import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import mup
import numpy as np

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi
from optimizer import DPISpectralOptimizer

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
    def encode(self, text):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
        return tokens

class TinyWiki(Dataset):
    def __init__(self, target_tokens=500_000):
        self.seq_len = 128
        tokenizer = SimpleBPETokenizer(16384)
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        all_tokens = []
        for entry in dataset:
            all_tokens.extend(tokenizer.encode(entry["text"]))
            if len(all_tokens) >= target_tokens: break
        self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def get_top_vectors(seed, mode, loader, device, base_model, steps=500):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=16384, d_model=320, n_heads=10, d_mlp=1280, n_layers=6)
    model = PID8Transformer(**cfg).to(device)
    mup.set_base_shapes(model, base_model)
    
    if mode == "dpi":
        initialize_dpi(model, loader, mode="v17.0")
        opt = DPISpectralOptimizer(model.parameters(), lr=8e-4, anchor_factor=0.42)
    else:
        for p in model.parameters():
            if p.dim() >= 2: nn.init.xavier_uniform_(p)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4)
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x).view(-1, 16384), y.view(-1))
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        W = model.layers[3].attn.W_q.weight.data
        U, S, V = torch.svd(W)
        return V[:, :5] # Top 5 semantic directions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🛰️ LEVEL 3: SINGULAR VECTOR STABILITY DUEL")
    loader = DataLoader(TinyWiki(), batch_size=32, shuffle=True)
    base_model = PID8Transformer(vocab_size=16384, d_model=64, n_heads=4, d_mlp=256, n_layers=6).to(device)
    
    print("\n--- Training DPI Models (Seeds 42 vs 1337) ---")
    V_dpi_1 = get_top_vectors(42, "dpi", loader, device, base_model)
    V_dpi_2 = get_top_vectors(1337, "dpi", loader, device, base_model)
    
    print("\n--- Training Xavier Models (Seeds 42 vs 1337) ---")
    V_xav_1 = get_top_vectors(42, "xavier", loader, device, base_model)
    V_xav_2 = get_top_vectors(1337, "xavier", loader, device, base_model)
    
    print("\n" + "="*60)
    print(f"{'Vector':<10} | {'DPI Stability':<15} | {'Xavier Stability'}")
    print("-" * 60)
    for i in range(5):
        sim_dpi = torch.abs(torch.cosine_similarity(V_dpi_1[:, i], V_dpi_2[:, i], dim=0)).item()
        sim_xav = torch.abs(torch.cosine_similarity(V_xav_1[:, i], V_xav_2[:, i], dim=0)).item()
        print(f"V_{i:<8} | {sim_dpi:.4f}          | {sim_xav:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
