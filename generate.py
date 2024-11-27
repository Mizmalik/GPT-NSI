from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from model import *
from config import *
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas.")
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    return model

def correct_text(text):
    blob = TextBlob(text)
    return str(blob.correct())

def translate_text(text, target_language):
    blob = TextBlob(text)
    try:
        return str(blob.translate(to=target_language))
    except Exception as e:
        return text

def generate_text(model, context, max_new_tokens=100):
    model.to(device)
    context_indices = [char_to_idx.get(c, char_to_idx[' ']) for c in context]
    context_tensor = torch.tensor(context_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(context_tensor, max_new_tokens=max_new_tokens)
    generated_text = ''.join([idx_to_char.get(i, '?') for i in generated_indices[0].tolist()])
    return generated_text

model_path = os.path.join(output_dir, model_filename)
model = load_model(model_path)

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
        return jsonify({"error": f"Caractère non reconnu dans l'entrée : {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur inattendue : {str(e)}"}), 500

app.run(debug=True)
