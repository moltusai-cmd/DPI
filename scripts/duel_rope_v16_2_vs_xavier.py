import torch
import torch.nn as nn
import sys
import os
import time
from torch.utils.data import DataLoader, Dataset, Subset
import math
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

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

def run_session(mode, loader, val_loader, device, total_steps=20000):
    print(f"\n🚀 Starting Marathon: RoPE + {mode.upper()} ({total_steps} steps)")
    torch.manual_seed(42)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8, use_rope=True).to(device)
    
    if mode == "xavier":
        print("  Applying Xavier Uniform Baseline...")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        warmup_steps = 400 # 2% of 20000
    else:
        print("  Applying DPI v16.2 Optimized...")
        initialize_dpi(model, loader, mode="v16.2", spectral_gamma=0.25)
        warmup_steps = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    it = iter(loader)
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        if step <= warmup_steps:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4 * (step / warmup_steps)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); optimizer.step()
        
        # Logging plus fréquent pour analyse de pente
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "loss": round(loss.item(), 4)})
            if step % 1000 == 0:
                print(f"  > Step {step:5d} | Train Loss: {loss.item():.4f}")
            
    return history

def analyze_slopes(res_xavier, res_dpi, target_loss=7.5):
    print(f"\n--- Delta-Slope Analysis at Target Loss: {target_loss} ---")
    
    def get_slope_at_target(history, target):
        # On cherche le moment où la loss traverse le target
        for i in range(1, len(history)):
            l1, l2 = history[i-1]["loss"], history[i]["loss"]
            if (l1 >= target and l2 <= target) or (l1 <= target and l2 >= target):
                s1, s2 = history[i-1]["step"], history[i]["step"]
                # Pente locale : (L2 - L1) / (S2 - S1)
                slope = (l2 - l1) / (s2 - s1)
                return slope, s2
        return None, None

    slope_x, step_x = get_slope_at_target(res_xavier, target_loss)
    slope_d, step_d = get_slope_at_target(res_dpi, target_loss)

    if slope_x and slope_d:
        ratio = slope_d / slope_x
        print(f"Xavier reached {target_loss} at step {step_x} | Slope: {slope_x:.6f}")
        print(f"DPI    reached {target_loss} at step {step_d} | Slope: {slope_d:.6f}")
        print(f"Learning Rate Acceleration Factor: {ratio:.2f}x")
    else:
        print("Target loss not reached by both models in this window.")

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    res_xavier = run_session("xavier", train_loader, val_loader, device, total_steps=100000)
    torch.cuda.empty_cache(); time.sleep(2)
    res_dpi = run_session("v16.2", train_loader, val_loader, device, total_steps=100000)
    
    analyze_slopes(res_xavier, res_dpi, target_loss=7.5)
    
    results = {"xavier": res_xavier, "dpi": res_dpi}
    with open("marathon_100k_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
