import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import copy
import mup
import numpy as np
import random

# Add src to path
sys.path.append('src')
from model import PID8Transformer, count_parameters
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

class RobustDataset(Dataset):
    def __init__(self, split="train", vocab_size=16384, seq_len=128, target_tokens=1_000_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        cache_path = f"results/tokens_cache_{split}_{target_tokens}.pt"
        if os.path.exists(cache_path):
            self.tokens = torch.load(cache_path)
        else:
            print(f"📦 Tokenizing {split} split...")
            tokenizer = SimpleBPETokenizer(vocab_size)
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
            all_tokens = []
            for entry in dataset:
                all_tokens.extend(tokenizer.encode(entry["text"]))
                if len(all_tokens) >= target_tokens: break
            self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
            os.makedirs("results", exist_ok=True)
            torch.save(self.tokens, cache_path)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def train_model(name, model, loader, val_loader, device, opt_type="AdamW", total_steps=1000, lr=8e-4):
    model.train()
    if opt_type == "DSO":
        # DSO v1.4 - Iron Anchor @ 1e-3
        optimizer = DPISpectralOptimizer(model.parameters(), lr=lr, weight_decay=0.01, anchor_factor=1.5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    steps = 0
    while steps < total_steps:
        for x, y in loader:
            if steps >= total_steps: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            if steps % 250 == 0:
                print(f"  [{name}] Step {steps:4d} | Loss: {loss.item():.4f}")
            if torch.isnan(loss):
                print(f"  [{name}] 💥 CRASHED at step {steps}")
                return 99.99
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return val_loss / len(val_loader)

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, seq_len = 16384, 128
    
    train_dataset = RobustDataset(split="train", target_tokens=1_000_000)
    val_dataset = RobustDataset(split="validation", target_tokens=100_000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=10, d_mlp=1280, n_layers=6, max_len=seq_len)
    base_cfg = dict(vocab_size=vocab_size, d_model=64, n_heads=4, d_mlp=256, n_layers=6, max_len=seq_len)
    base_model = PID8Transformer(**base_cfg).to(device)
    
    print(f"⚔️ ULTIMATE DUEL: AdamW (Standard) vs DSO v1.4 (IRON ANCHOR)")
    print(f"{'='*80}")

    # 1. AdamW @ 8e-4
    print("\n🚀 Running AdamW @ 8e-4 (Standard)...")
    m_adam = PID8Transformer(**cfg).to(device)
    mup.set_base_shapes(m_adam, base_model)
    initialize_dpi(m_adam, train_loader, mode="v16.3")
    loss_adam = train_model("AdamW@8e-4", m_adam, train_loader, val_loader, device, opt_type="AdamW", lr=8e-4)

    # 2. DSO v1.4 @ 1e-3 (Iron Anchor)
    # We aim to beat the 5.7394 record.
    print("\n🚀 Running DSO v1.4 @ 1e-3 (Iron Anchor - No Warmup)...")
    m_dso = PID8Transformer(**cfg).to(device)
    mup.set_base_shapes(m_dso, base_model)
    initialize_dpi(m_dso, train_loader, mode="v16.3")
    loss_dso = train_model("DPI+DSO", m_dso, train_loader, val_loader, device, opt_type="DSO", lr=1e-3)

    print(f"\n{'='*60}")
    print(f"{'Configuration':<30} | {'Val Loss':<10}")
    print(f"{'-'*60}")
    print(f"{'AdamW @ 8e-4':<30} | {loss_adam:<10.4f}")
    print(f"{'DSO v1.4 @ 1e-3 (Iron Anchor)':<30} | {loss_dso:<10.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
