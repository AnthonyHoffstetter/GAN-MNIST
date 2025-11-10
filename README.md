# GAN-MNIST
## Description

Ce projet implémente un **Generative Adversarial Network (GAN)** pour générer des images de chiffres manuscrits à partir du dataset MNIST.

Le GAN se compose de :

- **Generator** : crée des images à partir d’un vecteur aléatoire (bruit latent).
- **Discriminator** : évalue si une image est réelle ou générée.

Le projet inclut également un **OCR (EasyOCR)** pour reconnaître automatiquement les chiffres générés, montrant ainsi le “réalisme” des images produites.

## Fonctionnalités

- Entraînement complet du GAN sur MNIST.
- Affichage périodique des images générées pour suivre l’évolution de l’entraînement.
- Reconnaissance automatique du chiffre généré via OCR toutes les 10 époques.
- Possibilité de sauvegarder et recharger les modèles pour poursuivre l’entraînement.

## Installation

1. Cloner le projet :
    ```
    git clone https://github.com/AnthonyHoffstetter/GAN-MNIST/  
    cd GAN-MNIST
    ```

3. Installer les dépendances :
    ```
   pip install -r requirements.txt
    ```

4. (Optionnel) Installer **CUDA** pour utiliser le GPU pour accélérer l’entraînement.

## Utilisation

1. Lancer l’entraînement :
   ```
    python main.py
   ```

2. Pendant l'entraînement :

   - Le modèle affichera les pertes du Discriminator et du Generator à chaque époque.
   - Toutes les 10 époques, une image générée sera affichée avec le chiffre reconnu par l’OCR.

3. Les modèles sont sauvegardés automatiquement (si cette fonctionnalité est activée dans le code) pour pouvoir reprendre l’entraînement plus tard.
   
2. Astuces pour l'entraînement :

   - Il est possible de **commenter la partie affichage et test OCR** (ligne 160 à 171) pour accélérer l’entraînement.
   - Vous pouvez **augmenter le nombre d'époques** afin d'entraîner le GAN plus longtemps et obtenir des images de meilleure qualité.
   - En cas de crash ou de ralentissements, vous pouvez réduire le batch_size à 128 ou 64 (à modifier à la ligne 27).


## Organisation du projet

```text
├── main.py           # Code principal du GAN
├── data/             # Dataset MNIST téléchargé automatiquement
├── saved_models/     # Modèles sauvegardés
├── README.md
├── requirements.txt
```

## Résultats
Exemple de chiffre généré par le GAN et reconnu par l'OCR :

<img width="359" height="359" alt="image" src="https://github.com/user-attachments/assets/8001ce8d-cabc-4ce0-9d32-b9916ec7c47c" />

## A propos
Ce projet est réalisé dans le cadre du **Master 1 MIAGE** pour apprendre le fonctionnement des GANs et l’intégration d’un OCR sur les images générées.


