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

def train_model(name, model, loader, val_loader, device, total_steps=2000, lr=8e-4, use_mup_opt=False):
    model.train()
    if use_mup_opt:
        optimizer = mup.MuAdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 0% Warmup as per DPI protocol
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
    
    train_dataset = RobustDataset(split="train", target_tokens=1_000_000) # Fast ablation
    val_dataset = RobustDataset(split="validation", target_tokens=100_000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    cfg_base = dict(vocab_size=vocab_size, d_model=320, n_heads=10, d_mlp=1280, n_layers=6, max_len=seq_len)
    
    # Base shapes for muP
    base_model = PID8Transformer(vocab_size=vocab_size, d_model=64, n_heads=4, d_mlp=256, n_layers=6, max_len=seq_len).to(device)

    ablation_configs = [
        ("DPI PURE",          {"use_mup_attn": False, "use_mup_readout": False, "use_mup_opt": False}),
        ("DPI + MU_ATTN",     {"use_mup_attn": True,  "use_mup_readout": False, "use_mup_opt": False}),
        ("DPI + MU_OPT",      {"use_mup_attn": False, "use_mup_readout": False, "use_mup_opt": True}),
        ("DPI + MU_FULL",     {"use_mup_attn": True,  "use_mup_readout": True,  "use_mup_opt": True}),
    ]

    print(f"🔬 ABLATION STUDY: Why does muP-Fusion underperform DPI?")
    print(f"{'='*80}")
    
    results = []
    for name, flags in ablation_configs:
        print(f"\n🚀 Running {name}...")
        m = PID8Transformer(**cfg_base, use_mup_attn=flags["use_mup_attn"], use_mup_readout=flags["use_mup_readout"]).to(device)
        mup.set_base_shapes(m, base_model)
        
        # All use DPI v16.3 initialization
        initialize_dpi(m, train_loader, mode="v16.3")
        
        loss = train_model(name, m, train_loader, val_loader, device, total_steps=2000, lr=8e-4, use_mup_opt=flags["use_mup_opt"])
        results.append((name, loss))

    print(f"\n{'='*60}")
    print(f"{'Configuration':<25} | {'Val Loss':<10}")
    print(f"{'-'*60}")
    for name, loss in results:
        print(f"{name:<25} | {loss:<10.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
