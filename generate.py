from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from model import *  # Import du modèle défini séparément
from config import *  # Configuration du modèle et des paramètres
from textblob import TextBlob  # Pour correction et traduction

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes depuis d'autres domaines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choix du GPU si disponible

# Chargement du modèle
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas.")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# Correction du texte
def correct_text(text):
    return str(TextBlob(text).correct())

# Traduction du texte
def translate_text(text, target_language):
    try:
        return str(TextBlob(text).translate(to=target_language))
    except Exception:
        return text  # Retourne le texte original en cas d'échec

# Génération de texte à partir du modèle
def generate_text(model, context, max_new_tokens=100):
    model.to(device)
    context_indices = [char_to_idx.get(c, char_to_idx[' ']) for c in context]
    context_tensor = torch.tensor(context_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(context_tensor, max_new_tokens=max_new_tokens)
    return ''.join([idx_to_char.get(i, '?') for i in generated_indices[0].tolist()])

# Charger le modèle depuis le chemin spécifié
model_path = os.path.join(output_dir, model_filename)
model = load_model(model_path)

# Endpoint pour générer du texte
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    input_text = data.get("input", "")
    target_language = data.get("language", "fr")
    if not input_text:
        return jsonify({"error": "Texte d'entrée manquant"}), 400

    try:
        generated_text = generate_text(model, input_text, max_token)
        corrected_text = correct_text(generated_text)
        translated_text = translate_text(corrected_text, target_language)
        return jsonify({"generated": translated_text})
    except KeyError as e:
        return jsonify({"error": f"Caractère non reconnu : {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur : {str(e)}"}), 500

# Lancer le serveur Flask
app.run(debug=True)
