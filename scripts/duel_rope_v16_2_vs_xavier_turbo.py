import torch
import torch.nn as nn
import sys
import os
import time
from torch.utils.data import DataLoader, Dataset, Subset
import math
import json
import numpy as np

# Performance Boosters
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

def run_session(mode, loader, val_loader, device, total_steps=100000):
    print(f"\n🚀 Starting FULL RESOLUTION Turbo Marathon: RoPE + {mode.upper()} ({total_steps} steps)")
    torch.manual_seed(42)
    
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8, use_rope=True).to(device)
    
    if mode == "xavier":
        print("  Applying Xavier Uniform Baseline...")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        warmup_steps = 400 
    else:
        print("  Applying DPI v16.2 Optimized...")
        initialize_dpi(model, loader, mode="v16.2", spectral_gamma=0.25)
        warmup_steps = 0

    try:
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)
    except Exception as e:
        print(f"  Compilation skipped/failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = [] # On va tout stocker
    it = iter(loader)
    
    start_time = time.time()
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        if step <= warmup_steps:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4 * (step / warmup_steps)

        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # SAUVEGARDE DE CHAQUE POINT
        history.append(round(loss.item(), 5))
        
        if step % 5000 == 0:
            elapsed = time.time() - start_time
            sps = step / elapsed
            print(f"  > Step {step:6d} | Train Loss: {loss.item():.4f} | Speed: {sps:.1f} steps/sec")
            
    return history

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True, pin_memory=True)
    
    # Run Xavier
    res_xavier = run_session("xavier", train_loader, None, device, total_steps=100000)
    
    # Run DPI
    torch.cuda.empty_cache(); time.sleep(5)
    res_dpi = run_session("v16.2", train_loader, None, device, total_steps=100000)
    
    # Sauvegarde exhaustive
    results = {
        "metadata": {"total_steps": 100000, "arch": "20M-RoPE", "optim": "AdamW-Turbo"},
        "xavier": res_xavier, 
        "dpi": res_dpi
    }
    
    output_path = "marathon_100k_holy_grail.json"
    print(f"\nWriting {len(res_xavier) + len(res_dpi)} points to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f)
    
    print("\n🏁 HOLY GRAIL DATA COLLECTION COMPLETE.")

if __name__ == "__main__":
    main()
