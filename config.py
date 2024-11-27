import torch
import cpuinfo


# Définition des hyperparamètres du modèle et du processus d'entraînement
batch_size = 32 
sequence_length = 64  
num_iterations = 5000  
eval_interval = 250  
learning_rate = 1e-4  
output_dir = "trained_model"  
dataset_name = "simpsons"  
dataset_dir = "dataset"
dataset = dataset_dir + "/" + dataset_name + ".txt"  
model_filename = dataset_name + "-L.pth"  
random_seed = 1337  
eval_iterations = 200  
embedding_size = 64  
num_heads = 4  
num_layers = 4  
dropout_rate = 0.2  
max_token = 2000

cpu_info = cpuinfo.get_cpu_info()  # Information sur le CPU utilisé


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Chargement du texte d'entrée (exemple avec un fichier txt)
with open(dataset, encoding='utf-8') as f:
    text = f.read()

# Création du vocabulaire unique à partir du texte
unique_characters = sorted(list(set(text)))
vocab_size = len(unique_characters)

# Mappage des caractères vers des indices
char_to_idx = {ch: i for i, ch in enumerate(unique_characters)}  
idx_to_char = {i: ch for i, ch in enumerate(unique_characters)}  

# Encodage et décodage du texte
encode_text = lambda s: [char_to_idx[c] for c in s]  
decode_text = lambda l: ''.join([idx_to_char[i] for i in l])  

# Encodage du texte en indices
encoded_data = torch.tensor(encode_text(text), dtype=torch.long)

# Séparation en données d'entraînement et de validation (90% train, 10% val)
split_index = int(0.9 * len(encoded_data))
train_data = encoded_data[:split_index]
val_data = encoded_data[split_index:]