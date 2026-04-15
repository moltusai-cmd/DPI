import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import copy
import json
import argparse
import mup
import numpy as np
import random

# Add src to path for imports
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
            try:
                from datasets import load_dataset
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
                all_tokens = []
                for entry in dataset:
                    all_tokens.extend(tokenizer.encode(entry["text"]))
                    if len(all_tokens) >= target_tokens: break
                self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
                os.makedirs("results", exist_ok=True)
                torch.save(self.tokens, cache_path)
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Could not load WikiText-103 split '{split}'. Reason: {e}")
                sys.exit(1)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def calculate_stable_rank(model, threshold=0.01):
    """Computes the stable rank (isometry metric) of the Attention Query weights with a noise threshold."""
    with torch.no_grad():
        # Check Layer 3 W_q for consistency with previous benchmarks
        W = model.layers[3].attn.W_q.weight.data
        _, S, _ = torch.svd(W)
        # Filter out singular values below 1% of the maximum (noise floor)
        S_filtered = S[S > (threshold * S[0])]
        return (torch.sum(S_filtered**2) / (S[0]**2)).item()

def train_model(name, model, loader, val_loader, device, total_steps=2000, lr=1e-4, sched_type="Cosine", use_mup=False):
    model.train()
    optimizer = mup.MuAdamW(model.parameters(), lr=lr, weight_decay=0.01) if use_mup else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = 0 if ("DPI" in name) else 20
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        if sched_type == "Fixed": return 1.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
            scheduler.step()
            steps += 1
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return val_loss / len(val_loader)

def main():
    # SEED FOR REPRODUCIBILITY
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, seq_len = 16384, 128
    
    # Giga-Benchmark Standard: 40M tokens for training, 1M for validation
    train_dataset = RobustDataset(split="train", target_tokens=40_000_000)
    val_dataset = RobustDataset(split="validation", target_tokens=1_000_000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=10, d_mlp=1280, n_layers=6, max_len=seq_len, use_mup_attn=True, use_mup_readout=True)
    base_cfg = dict(vocab_size=vocab_size, d_model=64, n_heads=4, d_mlp=256, n_layers=6, max_len=seq_len, use_mup_attn=True, use_mup_readout=True)
    base_model = PID8Transformer(**base_cfg).to(device)
    
    print(f"Giga-Benchmark: 16 Tests (DPI vs muP) | Seed: {seed} | Device: {device}")
    all_results = []

    for init_name in ["Xavier Uniform", "True muP (MS)", "Pure DPI v16.2", "DPI-muP Fusion"]:
        for lr in [1e-4, 8e-4]:
            for sched in ["Cosine", "Fixed"]:
                display_name = f"{init_name} [{sched} @ {lr}]"
                print(f"Running {display_name}...")
                
                m = PID8Transformer(**cfg).to(device)
                mup.set_base_shapes(m, base_model)
                
                use_mup_opt = False
                start_init = time.time()
                
                if init_name == "Xavier Uniform":
                    for p in m.parameters(): 
                        if p.dim() >= 2: nn.init.xavier_uniform_(p)
                elif init_name == "True muP (MS)":
                    for p in m.parameters():
                        if p.dim() >= 2: mup.init.normal_(p, std=0.02)
                    use_mup_opt = True
                elif init_name == "Pure DPI v16.2":
                    initialize_dpi(m, train_loader, mode="v16.2")
                elif init_name == "DPI-muP Fusion":
                    initialize_dpi(m, train_loader, mode="v16.3")
                    use_mup_opt = True
                
                init_time = time.time() - start_init
                rank = calculate_stable_rank(m)
                
                start_train = time.time()
                loss = train_model(init_name, m, train_loader, val_loader, device, lr=lr, sched_type=sched, use_mup=use_mup_opt)
                train_time = time.time() - start_train
                
                all_results.append((init_name, f"{sched} @ {lr}", loss, rank, init_time, train_time))

    print(f"\n{'='*120}")
    print(f"{'Initialization':<20} | {'Regime':<18} | {'Val Loss':<10} | {'Rank':<8} | {'Init(s)':<8} | {'Train(s)':<8} | {'Advantage'}")
    print(f"{'-'*120}")
    base_loss = all_results[0][2]
    for name, sched, loss, rank, t_init, t_train, in all_results:
        adv = base_loss - loss
        print(f"{name:<20} | {sched:<18} | {loss:<10.4f} | {rank:<8.2f} | {t_init:<8.2f} | {t_train:<8.1f} | {adv:<10.4f}")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()
