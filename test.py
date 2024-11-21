import torch
import torch.nn as nn
import os
from torch.nn import functional as F
import cpuinfo
from tqdm import tqdm


# Définition des hyperparamètres du modèle et du processus d'entraînement
batch_size = 64  
sequence_length = 128  
num_iterations = 5000  
eval_interval = 250  
learning_rate = 1e-4  
output_dir = "trained_model"  
dataset_name = "simpsons"  
model_filename = dataset_name + "-L.pth"  
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
random_seed = 1337  
eval_iterations = 200  
embedding_size = 64  
num_heads = 4  
num_layers = 4  
dropout_rate = 0.0  

cpu_info = cpuinfo.get_cpu_info()  # Information sur le CPU utilisé

# Initialisation de la graine pour la reproductibilité
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)  
    print(f"Seed CUDA : {random_seed}")
    print(f"GPU : {torch.cuda.get_device_name(0)}")  
else:
    torch.manual_seed(random_seed)  
    print(f"Seed CPU : {random_seed}")
    print(f"CPU : {cpu_info['brand_raw']}")  

# Chargement du texte d'entrée (exemple avec un fichier txt)
with open(dataset_name + '.txt', encoding='utf-8') as f:
    text = f.read()

os.makedirs(output_dir, exist_ok=True)  # Création du dossier de sauvegarde
print(f"Le dossier {output_dir} a été créé")

# Affichage du nombre de tokens (caractères uniques) dans le texte
print(f"Nombre total de tokens : {len(text)}")

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

# Définition de la classe Head (tête d'attention)
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length)))  
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape  
        key = self.key(x)  
        query = self.query(x)  
        attention_weights = query @ key.transpose(-2, -1) * embedding_dim ** -0.5  
        attention_weights = attention_weights.masked_fill(self.tril[:seq_length, :seq_length] == 0, float('-inf'))  
        attention_weights = F.softmax(attention_weights, dim=-1)  
        attention_weights = self.dropout(attention_weights)  
        value = self.value(x)  
        output = attention_weights @ value  
        return output

# Définition de la classe MultiHeadAttention qui combine plusieurs têtes d'attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)  
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, x):
        output = torch.cat([head(x) for head in self.attention_heads], dim=-1)  
        output = self.dropout(self.proj(output))  
        return output

# Définition de la classe FeedForward (réseau feedforward)
class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),  
            nn.ReLU(),  
            nn.Linear(4 * embedding_size, embedding_size),  
            nn.Dropout(dropout_rate),  
        )

    def forward(self, x):
        return self.net(x)

# Définition du bloc Transformer (composé d'attention et de feedforward)
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        head_size = embedding_size // num_heads  
        self.self_attention = MultiHeadAttention(num_heads, head_size)  
        self.feedforward = FeedForwardLayer(embedding_size)  
        self.layer_norm1 = nn.LayerNorm(embedding_size)  
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))  
        x = x + self.feedforward(self.layer_norm2(x))  
        return x

# Modèle de langage basé sur un Transformer
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)  
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)  
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_size, num_heads) for _ in range(num_layers)])  
        self.layer_norm = nn.LayerNorm(embedding_size)  
        self.output_head = nn.Linear(embedding_size, vocab_size)  

    def forward(self, idx, targets=None):
        batch_size, seq_length = idx.shape
        token_embeddings = self.token_embedding(idx)  
        position_embeddings = self.position_embedding(torch.arange(seq_length, device=device))  
        x = token_embeddings + position_embeddings  
        x = self.transformer_blocks(x)  
        x = self.layer_norm(x)  
        logits = self.output_head(x)  

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -sequence_length:]  
            logits, _ = self(idx_cond)  
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)  
            next_token_idx = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, next_token_idx), dim=1)  
        return idx

# Initialisation et déplacement du modèle sur le device
language_model = TransformerLanguageModel()
model = language_model.to(device)
print(f"Device utilisé : {device}")
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

# Génération de texte
context = torch.zeros((1, 1), dtype=torch.long, device=device)  
generated_text = decode_text(model.generate(context, max_new_tokens=2000)[0].tolist())  
print(generated_text)
