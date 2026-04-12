import torch
import torch.nn as nn
import sys
import os
import time
from torch.utils.data import DataLoader, Dataset, Subset
import math
import random
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def run_session(mode, loader, val_loader, device, total_steps=2000):
    print(f"\n🚀 Starting Session: {mode.upper()}")
    set_seed(42)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if mode == "xavier":
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        warmup_steps = 140
    else: # v16
        initialize_dpi(model, loader, mode="v16", mlp_jitter=0.02)
        warmup_steps = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine scheduler for both
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, 7000 - warmup_steps))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    history = {}
    it = iter(loader)
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); optimizer.step(); scheduler.step()
        
        if step % 500 == 0 or step == total_steps:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                val_it = iter(val_loader)
                for _ in range(20):
                    vx, vy = next(val_it)
                    v_logits = model(vx.to(device))
                    v_loss += criterion(v_logits.view(-1, v_logits.size(-1)), vy.to(device).view(-1)).item()
            avg_v = v_loss / 20
            history[step] = avg_v
            print(f"  > Step {step:4d} | Val Loss: {avg_v:.4f}")
            model.train()
    return history

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    res_xavier = run_session("xavier", train_loader, val_loader, device, total_steps=2000)
    torch.cuda.empty_cache(); time.sleep(2)
    res_v16 = run_session("v16", train_loader, val_loader, device, total_steps=2000)
    
    print("\n" + "="*50)
    print("🏆 FINAL DUEL: XAVIER vs DPI v16.0")
    print("="*50)
    print(f"{'Step':<6} | {'Xavier Baseline':<15} | {'DPI v16.0 (Phase)'} | {'Delta'}")
    print("-" * 50)
    for s in [500, 1000, 1500, 2000]:
        diff = res_v16[s] - res_xavier[s]
        print(f"{s:<6} | {res_xavier[s]:.4f}          | {res_v16[s]:.4f}          | {diff:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
