"""
Transformer-based Mental Health Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, n_embd, num_heads, dropout=0.1):
        super().__init__()
        assert n_embd % num_heads == 0
        
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out = nn.Linear(n_embd, n_embd)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, n_embd = x.size()
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        output = self.out(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, n_embd, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.ln1(x), attention_mask)
        x = x + self.dropout(attn_output)
        
        # MLP with residual connection
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        
        return x


class MentalHealthClassifier(nn.Module):
    """
    Transformer-based mental health classifier.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model parameters
        vocab_size = config.get('vocab_size', 10000)
        n_embd = config.get('n_embd', 256)
        num_heads = config.get('num_heads', 4)
        n_layer = config.get('n_layer', 3)
        num_classes = config.get('num_classes', 3)
        max_seq_length = config.get('max_seq_length', 256)
        dropout = config.get('dropout', 0.3)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_length, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(n_embd, num_heads, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layers
        self.ln_final = nn.LayerNorm(n_embd)
        self.classifier = nn.Linear(n_embd, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combined embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Global average pooling (masked)
        if attention_mask is not None:
            # Expand attention mask for broadcasting
            attention_mask = attention_mask.unsqueeze(-1).float()
            x = x * attention_mask
            pooled = x.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'hidden_states': x,
            'pooled_output': pooled
        }


def create_model(config):
    """
    Factory function to create a mental health classifier model.
    
    Args:
        config (dict): Model configuration
        
    Returns:
        MentalHealthClassifier: Initialized model
    """
    return MentalHealthClassifier(config)
