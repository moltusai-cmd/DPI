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

# Add src to path for core logic
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. REPRODUCIBILITY SETUP ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. DATASET LOADER ---
class FastWikiDataset(Dataset):
    def __init__(self, cache_path, seq_len=128):
        self.seq_len = seq_len
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file {cache_path} not found.")
        self.data = torch.load(cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

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

# --- 3. TRAINING ENGINE ---
def run_single_experiment(seed, mode, warmup_pct, device, full_dataset, indices, split, total_steps=7000):
    set_seed(seed)
    print(f"\n[RUN] Seed: {seed:4d} | Mode: {mode:6s} | Warmup: {warmup_pct*100:.1f}%")
    
    # 3.1 Per-run DataLoader with Seeded Generator
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        Subset(full_dataset, indices[:split]), 
        batch_size=32, 
        shuffle=True, 
        generator=g
    )
    val_loader = DataLoader(
        Subset(full_dataset, indices[split:]), 
        batch_size=32, 
        shuffle=False
    )
    
    # Model Config (20M Scale)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    # Initialization
    if mode == "dpi":
        initialize_dpi(model, train_loader, warp_zeta=1.1, spectral_gamma=0.25, use_calibration=True)
    else:
        xavier_init(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Warmup calculation
    warmup_steps = int(warmup_pct * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    milestones = [500, 2000, 7000]
    
    model.train()
    step = 0
    
    # Standard training loop
    it = iter(train_loader)
    while step < total_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step in milestones:
            val_loss = evaluate(model, val_loader, device)
            results[step] = val_loss
            print(f"  > Step {step:4d} | Val Loss: {val_loss:.4f}")
            model.train()
            
    return results

# --- 4. MAIN EXPERIMENT SUITE ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    seeds = [42, 123, 7, 2024, 99]
    output_dir = "experiments/MultiSeed_Robustness"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Dataset
    full_dataset = FastWikiDataset(cache_file)
    indices = list(range(len(full_dataset)))
    split = int(0.9 * len(full_dataset))
    
    all_metrics = {"dpi": [], "xavier": []}
    
    # Execute All Runs
    for mode, warmup in [("dpi", 0.0), ("xavier", 0.02)]:
        for seed in seeds:
            res = run_single_experiment(seed, mode, warmup, device, full_dataset, indices, split)
            all_metrics[mode].append(res)
            
    # --- 5. STATISTICAL ANALYSIS ---
    summary = {}
    print("\n" + "="*50)
    print("🏆 MULTI-SEED ROBUSTNESS REPORT (N=5)")
    print("="*50)
    print(f"{'Step':<6} | {'DPI (μ ± σ)':<18} | {'Xavier (μ ± σ)':<18} | {'Overlap?'}")
    print("-" * 65)
    
    for step in [500, 2000, 7000]:
        dpi_vals = [m[step] for m in all_metrics["dpi"]]
        xav_vals = [m[step] for m in all_metrics["xavier"]]
        
        mu_dpi, std_dpi = np.mean(dpi_vals), np.std(dpi_vals)
        mu_xav, std_xav = np.mean(xav_vals), np.std(xav_vals)
        
        # Check for overlap (Simple Confidence Interval mu +- std)
        overlap = not (mu_dpi + std_dpi < mu_xav - std_xav or mu_xav + std_xav < mu_dpi - std_dpi)
        overlap_str = "YES (Wait)" if overlap else "NO (REAL SIGNAL)"
        
        print(f"{step:<6} | {mu_dpi:.4f} ± {std_dpi:.3f} | {mu_xav:.4f} ± {std_xav:.3f} | {overlap_str}")
        
        summary[step] = {
            "dpi": {"mean": round(float(mu_dpi), 4), "std": round(float(std_dpi), 4)},
            "xavier": {"mean": round(float(mu_xav), 4), "std": round(float(std_xav), 4)},
            "overlap": overlap
        }

    # Save final report
    final_output = {
        "metadata": {"seeds": seeds, "total_steps": 7000, "arch": "20M"},
        "summary": summary,
        "raw_data": all_metrics
    }
    
    with open(f"{output_dir}/multiseed_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("="*65)
    print(f"REPORT SAVED TO: {output_dir}/multiseed_results.json")

if __name__ == "__main__":
    main()
