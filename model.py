import torch
import torch.nn as nn
from torch.nn import functional as F

from config import *  # Importation des paramètres de configuration

# Classe pour une tête d'attention
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length)))  # Masque causal
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        key = self.key(x)
        query = self.query(x)
        attention_weights = query @ key.transpose(-2, -1) * x.shape[-1] ** -0.5  # Produit scalaire
        attention_weights = attention_weights.masked_fill(self.tril[:x.shape[1], :x.shape[1]] == 0, float('-inf'))  
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        value = self.value(x)
        return attention_weights @ value  # Application des poids sur les valeurs

# Classe pour plusieurs têtes d'attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Concaténation des sorties des têtes d'attention
        output = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        return self.dropout(self.proj(output))

# Classe pour le réseau feedforward
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

# Bloc Transformer combinant attention et feedforward
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        head_size = embedding_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feedforward = FeedForwardLayer(embedding_size)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Résidu + attention
        x = x + self.self_attention(self.layer_norm1(x))
        # Résidu + feedforward
        return x + self.feedforward(self.layer_norm2(x))

# Modèle de langage Transformer
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)  # Embedding des tokens
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)  # Embedding des positions
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_size, num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.output_head = nn.Linear(embedding_size, vocab_size)  # Logits pour le vocabulaire

    def forward(self, idx, targets=None):
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(torch.arange(idx.shape[1], device=device))
        x = token_embeddings + position_embeddings  # Somme des embeddings
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.output_head(x)

        # Calcul de la perte si les cibles sont fournies
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -sequence_length:]  # Contexte limité
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Dernier token
            probs = F.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token_idx), dim=1)  # Ajout du nouveau token
        return idx
