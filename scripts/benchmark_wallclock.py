import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. DATASET SETUP ---
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

# --- 2. BENCHMARK FUNCTIONS ---

def measure_init_time(method, model, loader, device):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    if method == "dpi":
        initialize_dpi(model, loader, warp_zeta=1.1, spectral_gamma=0.25, use_calibration=True)
    else:
        xavier_init(model)
        
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def measure_step_time(model, loader, device, num_steps=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # Warmup steps for GPU
    it = iter(loader)
    for _ in range(10):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for i in range(num_steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / num_steps

def find_steps_to_target(model, loader, target_loss, device, max_steps=10000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    step = 0
    it = iter(loader)
    while step < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        step += 1
        
        if loss.item() <= target_loss:
            return step
    return max_steps

# --- 3. MAIN RUNNER ---

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    target_loss = 6.5
    
    print(f"🚀 Starting Wall-Clock Benchmark (Target Loss: {target_loss})")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    results = {"xavier": [], "dpi": []}
    
    for method in ["xavier", "dpi"]:
        print(f"\n--- Method: {method.upper()} ---")
        init_times = []
        step_times = []
        steps_to_target_list = []
        
        for run in range(3):
            # Reset model and seed for each run
            torch.manual_seed(42)
            model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
            
            # 1. T_init
            t_init = measure_init_time(method, model, loader, device)
            init_times.append(t_init)
            
            # 2. T_step
            t_step = measure_step_time(model, loader, device)
            step_times.append(t_step)
            
            # 3. Steps to Target
            # (We only measure this once properly or average it, but here we do it per run)
            # To save time, we use a single run for steps_to_target but init/step for all 3
            if run == 0:
                print(f"  Measuring Steps to Target Loss {target_loss}...")
                s_target = find_steps_to_target(model, loader, target_loss, device)
                steps_to_target_list.append(s_target)
            
            print(f"  Run {run+1}: T_init={t_init:.3f}s, T_step={t_step:.4f}s")
            
        # Median calculations
        med_init = np.median(init_times)
        med_step = np.median(step_times)
        s_target = steps_to_target_list[0]
        
        t_total = med_init + (med_step * s_target)
        
        results[method] = {
            "T_init": med_init,
            "T_step": med_step,
            "Steps": s_target,
            "T_total": t_total
        }

    # Output Table
    print("\n" + "="*70)
    print(f"{'Method':<10} | {'T_init (s)':<12} | {'T_step (s)':<12} | {'Steps->6.5':<12} | {'T_total (s)'}")
    print("-" * 70)
    for m in ["xavier", "dpi"]:
        r = results[m]
        print(f"{m.upper():<10} | {r['T_init']:<12.3f} | {r['T_step']:<12.4f} | {r['Steps']:<12} | {r['T_total']:<12.2f}")
    
    advantage = results["xavier"]["T_total"] / results["dpi"]["T_total"]
    print("="*70)
    print(f"🏆 DPI Wall-Clock Advantage: {advantage:.2f}x faster to reach target loss.")

if __name__ == "__main__":
    main()
