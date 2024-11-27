import torch
import os
from tqdm import tqdm
                                                                                                    
from model import *
from config import *


print(f"Device utilisé : {device}") #Information sur le device utilisé (Si c'est un cpu ou gpu (cuda))

# Initialisation de la graine pour la reproductibilité
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)  
    print(f"Seed CUDA : {random_seed}")
    print(f"GPU : {torch.cuda.get_device_name(0)}")  
else:
    torch.manual_seed(random_seed)  
    print(f"Seed CPU : {random_seed}")
    print(f"CPU : {cpu_info['brand_raw']}")  

os.makedirs(output_dir, exist_ok=True)  # Création du dossier de sauvegarde
print(f"Le dossier {output_dir} a été créé")

# Affichage du nombre de tokens (caractères uniques) dans le texte
print(f"Nombre total de tokens : {len(text)}")

# Fonction pour obtenir un lot de données pour l'entraînement ou la validation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    random_indices = torch.randint(len(data) - sequence_length, (batch_size,))  
    input_sequence = torch.stack([data[i:i + sequence_length] for i in random_indices])  
    target_sequence = torch.stack([data[i + 1:i + sequence_length + 1] for i in random_indices])  
    input_sequence, target_sequence = input_sequence.to(device), target_sequence.to(device)  
    return input_sequence, target_sequence

# Fonction pour évaluer la perte sur l'ensemble de validation
@torch.no_grad()
def estimate_loss():
    loss_output = {}
    model.eval()  # Mise en mode évaluation pour désactiver dropout et batchnorm
    
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        # Pré-génération de tous les indices pour les lots
        indices = torch.randint(len(data) - sequence_length, (eval_iterations, batch_size))
        # Initialisation du total de perte
        total_loss = 0
        # Traitement par lots
        for batch_indices in indices:
            input_batch = torch.stack([data[i:i + sequence_length] for i in batch_indices]).to(device)
            target_batch = torch.stack([data[i + 1:i + sequence_length + 1] for i in batch_indices]).to(device)
            logits, loss = model(input_batch, target_batch)  
            total_loss += loss.item()
        # Moyenne des pertes sur les lots
        loss_output[split] = total_loss / eval_iterations
    model.train()  # Repassage en mode entraînement
    return loss_output

# Initialisation et déplacement du modèle sur le device
language_model = TransformerLanguageModel()
model = language_model.to(device)
print(f"Nombre total de paramètres : {sum(p.numel() for p in model.parameters()) / 1e6}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  

# Boucle d'entraînement avec barre de progression
with tqdm(range(num_iterations), desc="Entraînement", unit="itération") as progress_bar:
    for iteration in progress_bar:

        # Évaluation périodique des pertes
        if iteration % eval_interval == 0 or iteration == num_iterations - 1:
            loss_values = estimate_loss()  
            progress_bar.set_postfix({
                "Train Loss": f"{loss_values['train']:.4f}",
                "Val Loss": f"{loss_values['val']:.4f}"
            })  # Mise à jour des infos sur la barre de progression

        # Chargement d'un batch et calcul des pertes
        input_batch, target_batch = get_batch('train')  
        logits, loss = model(input_batch, target_batch)  
        optimizer.zero_grad(set_to_none=True)  
        loss.backward()  
        optimizer.step()

# Sauvegarde du modèle
torch.save(model, os.path.join(output_dir, model_filename))  
print(f"Le modèle a été sauvegardé sous le nom de {model_filename}")
