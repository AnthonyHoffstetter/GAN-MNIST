import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import easyocr
from PIL import Image
import os

# Initialisation OCR
reader = easyocr.Reader(['en'])

# Vérification GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil utilisé :", device)

# Transformation des images : redimension, normalisation (-1 à 1 pour tanh)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Chargement du dataset MNIST
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

latent_dim = 100  # taille du bruit d'entrée

# ---------------- Generator ----------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img

generator = Generator().to(device)

# ---------------- Discriminator ----------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

discriminator = Discriminator().to(device)

# ---------------- Loss et optim ----------------
criterion = nn.BCELoss()
lr = 0.0002
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# ---------------- OCR Helper ----------------
def ocr_digit(img_tensor):
    img = ((img_tensor.squeeze().cpu().numpy() + 1) * 127.5).astype(np.uint8)
    img_pil = Image.fromarray(img)
    result = reader.readtext(np.array(img_pil), detail=0)
    return result[0] if len(result) > 0 else "?"

# ---------------- Dossier pour sauvegarde ----------------
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# Charger modèles existants si présents
gen_path = os.path.join(save_dir, "generator.pth")
disc_path = os.path.join(save_dir, "discriminator.pth")

if os.path.exists(gen_path):
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    print("Générateur chargé depuis la dernière sauvegarde !")
if os.path.exists(disc_path):
    discriminator.load_state_dict(torch.load(disc_path, map_location=device))
    print("Discriminateur chargé depuis la dernière sauvegarde !")

# # --- Test rapide du générateur avec OCR ---
# generator.eval()

# # Génération d'une image
# z = torch.randn(1, latent_dim).to(device)
# fake_image = generator(z)

# # OCR
# recognized_digit = ocr_digit(fake_image.detach())
# print(f"Chiffre reconnu par l'OCR : {recognized_digit}")

# # Conversion pour affichage
# img_to_show = fake_image.detach().cpu().squeeze().numpy()
# plt.imshow((img_to_show + 1) / 2, cmap="gray")
# plt.title("Image générée")
# plt.show()



# ---------------- Training ----------------
epochs = 50

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)
        valid = torch.ones((batch_size, 1), device=device)
        fake = torch.zeros((batch_size, 1), device=device)

        # ----- Generator -----
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ----- Discriminator -----
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Époque [{epoch+1}/{epochs}]  |  D loss: {d_loss.item():.4f}  |  G loss: {g_loss.item():.4f}")

    # ----- Test OCR toutes les 10 époques et sauvegarde -----
    if (epoch + 1) % 10 == 0:
        generator.eval()
        with torch.no_grad():
            # Test image
            z_test = torch.randn(1, latent_dim, device=device)
            fake_img = generator(z_test)

            # OCR
            recognized_digit = ocr_digit(fake_img)
            print(f"Chiffre reconnu par l'OCR : {recognized_digit}")

            # Affichage
            plt.imshow((fake_img.squeeze().cpu().numpy() + 1)/2, cmap="gray")
            plt.title(f"Époque {epoch+1}")
            plt.show()

            # Sauvegarde
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            print(f"Modèles sauvegardés à l'époque {epoch+1}")

        generator.train()
