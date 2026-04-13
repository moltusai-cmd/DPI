import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        self.cache = None

    def forward(self, x, seq_len):
        if self.cache is not None and self.cache.shape[1] >= seq_len:
            return self.cache[:, :seq_len]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cache = emb[None, :, None, :] # [1, T, 1, C]
        return self.cache

def apply_rotary_pos_emb(x, cos, sin):
    # x shape: [B, T, H, D]
    dim = x.shape[-1]
    half_dim = dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    # rotate_half
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, rope=None):
        B, T, C = x.size()
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        if rope is not None:
            emb = rope(x, T) # [1, T, 1, D]
            cos = emb.cos(); sin = emb.sin()
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        # Use Flash Attention (scaled_dot_product_attention)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.W_o(out))

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_mlp, bias=False)
        self.W2 = nn.Linear(d_mlp, d_model, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.act(self.W1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, dropout)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope=rope)
        x = x + self.mlp(self.ln2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PID8Transformer(nn.Module):
    def __init__(self, vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8, max_len=512, dropout=0.1, use_rope=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(d_model // n_heads, max_len)
            self.pos_encoding = nn.Identity() # Bypassed
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_len)
            self.rope = None
            
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.gradient_checkpointing = False

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        if not self.use_rope:
            x = self.pos_encoding(x)
            
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope, use_reentrant=False)
            else:
                x = layer(x, rope=self.rope)
        x = self.ln_f(x)
        return self.unembed(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
