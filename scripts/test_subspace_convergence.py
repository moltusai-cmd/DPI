import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import numpy as np
import random
import mup

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi
from optimizer import DPISpectralOptimizer

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
    def encode(self, text, target_count=None):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if target_count and len(tokens) >= target_count: break
        return tokens

class TinyWiki(Dataset):
    def __init__(self, target_tokens=1_000_000):
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

def get_subspace(model):
    with torch.no_grad():
        W = model.layers[3].attn.W_q.weight.data
        U, S, V = torch.svd(W)
        return U[:, :10]

def calculate_alignment(U1, U2):
    alignment_matrix = torch.matmul(U1.t(), U2)
    singular_values = torch.svd(alignment_matrix).S
    return singular_values.mean().item()

def run_convergence_test(mode, loader, device, base_model, steps=[0, 500, 1000]):
    results = {}
    
    # Model 1 (Seed 42)
    torch.manual_seed(42)
    m1 = PID8Transformer(vocab_size=16384, d_model=320, n_heads=10, d_mlp=1280, n_layers=6).to(device)
    mup.set_base_shapes(m1, base_model)
    
    # Model 2 (Seed 1337)
    torch.manual_seed(1337)
    m2 = PID8Transformer(vocab_size=16384, d_model=320, n_heads=10, d_mlp=1280, n_layers=6).to(device)
    mup.set_base_shapes(m2, base_model)

    if mode == "dpi":
        initialize_dpi(m1, loader, mode="v16.3")
        initialize_dpi(m2, loader, mode="v16.3")
        opt1 = DPISpectralOptimizer(m1.parameters(), lr=8e-4, anchor_factor=0.42)
        opt2 = DPISpectralOptimizer(m2.parameters(), lr=8e-4, anchor_factor=0.42)
    else:
        for m in [m1, m2]:
            for p in m.parameters():
                if p.dim() >= 2: nn.init.xavier_uniform_(p)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=8e-4)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=8e-4)

    criterion = nn.CrossEntropyLoss()
    
    for s in range(max(steps) + 1):
        if s in steps:
            U1, U2 = get_subspace(m1), get_subspace(m2)
            results[s] = calculate_alignment(U1, U2)
            print(f"  [{mode.upper()}] Step {s:4d} | Alignment: {results[s]:.4f}")
        
        # Train one step
        for m, opt in [(m1, opt1), (m2, opt2)]:
            m.train()
            x, y = next(iter(loader))
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(m(x).view(-1, 16384), y.view(-1))
            loss.backward()
            opt.step()
            
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🛰️ LEVEL 4: PLATONIC SUBSPACE CONVERGENCE (Init -> 1000 Steps)")
    loader = DataLoader(TinyWiki(500_000), batch_size=32, shuffle=True)
    base_model = PID8Transformer(vocab_size=16384, d_model=64, n_heads=4, d_mlp=256, n_layers=6).to(device)
    
    print("\n--- Testing DPI Convergence ---")
    res_dpi = run_convergence_test("dpi", loader, device, base_model)
    
    print("\n--- Testing Xavier Convergence ---")
    res_xav = run_convergence_test("xavier", loader, device, base_model)
    
    print("\n" + "="*70)
    print(f"{'Step':<10} | {'DPI Alignment':<15} | {'Xavier Alignment':<15} | {'Delta'}")
    print("-" * 70)
    for s in [0, 500, 1000]:
        diff = res_dpi[s] - res_xav[s]
        print(f"{s:<10} | {res_dpi[s]:.4f}          | {res_xav[s]:.4f}             | {diff:+.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
