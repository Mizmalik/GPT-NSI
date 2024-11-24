import argparse
import torch
import os
from model import * 
from config import *
from textblob import TextBlob  # Bibliothèque pour la correction orthographique

def load_model(model_path):
    """Charge le modèle sauvegardé."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas.")
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()  # Met le modèle en mode évaluation
    return model

def correct_text(text):
    """Corrige le texte généré."""
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

def generate_text(model, context, max_new_tokens=100):
    """Génère du texte basé sur un contexte donné."""
    model.to(device)
    context_indices = [char_to_idx.get(c, char_to_idx[' ']) for c in context]  # Encode le contexte, remplace les inconnus par des espaces
    context_tensor = torch.tensor(context_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(context_tensor, max_new_tokens=max_new_tokens)
    generated_text = ''.join([idx_to_char.get(i, '?') for i in generated_indices[0].tolist()])
    return generated_text

# Définition des arguments de ligne de commande
parser = argparse.ArgumentParser(description="Générer du texte avec un modèle de langage pré-entraîné.")
parser.add_argument("--input", required=True, type=str, help="Texte d'entrée pour initialiser la génération.")
parser.add_argument("--max_tokens", required=True, type=int, help="Nombre maximal de tokens à générer.")
parser.add_argument("--model_dir", default=output_dir, type=str, help="Répertoire où le modèle est sauvegardé.")
parser.add_argument("--model_file", default=model_filename, type=str, help="Nom du fichier du modèle sauvegardé.")

args = parser.parse_args()

# Chargement du modèle
model_path = os.path.join(args.model_dir, args.model_file)
print(f"Chargement du modèle depuis {model_path}...")
try:
    model = load_model(model_path)
    print("Modèle chargé avec succès!")
except FileNotFoundError as e:
    print(e)
    exit(1)

# Génération et correction du texte
try:
    generated_text = generate_text(model, args.input, args.max_tokens)
    
    # Correction orthographique
    corrected_text = correct_text(generated_text)
    print(corrected_text)
except KeyError as e:
    print(f"Erreur : caractère non reconnu dans l'entrée. Détails : {e}")
