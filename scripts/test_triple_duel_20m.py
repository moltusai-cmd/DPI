import torch
import torch.nn as nn
import sys
import os
import time
import json
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import math

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. REPRODUCIBILITY ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class FastWikiDataset(Dataset):
    def __init__(self, cache_path, seq_len=128):
        self.seq_len = seq_len
        self.data = torch.load(cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return (torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long),
                torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long))

def evaluate(model, loader, device, max_steps=50):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_steps: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return round(total_loss / max_steps, 4)

# --- 2. TRAINING ENGINE ---
def run_session(seed, mode, loader, val_loader, device, total_steps=2000):
    set_seed(seed)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if mode == "xavier":
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        warmup_steps = 140
    elif mode == "gold":
        # DPI 14.1 (Linear Alignment)
        initialize_dpi(model, loader, use_attention_arch=False, mlp_jitter=0.02)
        warmup_steps = 0
    else:
        # DPI 15.2 (Attention Arch @ 0.40)
        initialize_dpi(model, loader, use_attention_arch=True, alignment_peak=0.40, mlp_jitter=0.02)
        warmup_steps = 0
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, 7000 - warmup_steps))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    it = iter(loader)
    for step in range(total_steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); optimizer.step(); scheduler.step()
        
    final_val = evaluate(model, val_loader, device)
    return final_val

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    seeds = [42, 123, 7]
    results = {"xavier": [], "gold": [], "hyper": []}
    
    for mode in ["xavier", "gold", "hyper"]:
        print(f"\n🚀 Testing Mode: {mode.upper()}")
        for s in seeds:
            res = run_session(s, mode, train_loader, val_loader, device)
            results[mode].append(res)
            print(f"  Seed {s} | Val Loss: {res:.4f}")
            torch.cuda.empty_cache()
            
    # --- REPORT ---
    print("\n" + "="*65)
    print(f"🏆 TRIPLE DUEL REPORT (2000 steps, N=3)")
    print("="*65)
    print(f"{'Mode':<15} | {'Mean Loss':<15} | {'Std Dev'}")
    print("-" * 65)
    for m in ["xavier", "gold", "hyper"]:
        mu, std = np.mean(results[m]), np.std(results[m])
        print(f"{m.upper():<15} | {mu:.4f}          | {std:.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
