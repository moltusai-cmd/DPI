import torch
import torch.nn as nn
import sys
import os
import time
import json
from torch.utils.data import DataLoader, Dataset, Subset
import math

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. DATASET ---
class FastWikiDataset(Dataset):
    def __init__(self, cache_path, seq_len=128):
        self.seq_len = seq_len
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

# --- 2. TRAINING ENGINE ---
def run_training(mode, device, loader, val_loader, total_steps=7000):
    torch.manual_seed(42) # Fixed seed for direct comparison
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if mode == "xavier":
        print(f"\n🚀 Mode: Xavier Baseline | Warmup: 2%")
        xavier_init(model)
        warmup_steps = 140
    else:
        print(f"\n🚀 Mode: DPI + 0.02 Jitter | Warmup: 0%")
        # Use our Gold Standard recipe: 0.02 jitter is now default in 14.1
        initialize_dpi(model, loader, use_exact_svd=True, mlp_jitter=0.02)
        warmup_steps = 0
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    history = {}
    milestones = [500, 2000, 7000]
    
    model.train()
    step = 0
    it = iter(loader)
    
    while step < total_steps:
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step in milestones:
            val_loss = evaluate(model, val_loader, device)
            history[step] = val_loss
            print(f"  > Step {step:4d} | Val Loss: {val_loss:.4f}")
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
    
    results_x = run_training("xavier", device, train_loader, val_loader)
    torch.cuda.empty_cache(); time.sleep(5)
    results_d = run_training("dpi_jitter", device, train_loader, val_loader)
    
    print("\n" + "="*50)
    print("🏆 OPTIMAL CONFIGURATION DUEL")
    print("="*50)
    print(f"{'Step':<6} | {'Xavier (Baseline)':<18} | {'DPI+Jitter (Optimal)'}")
    print("-" * 50)
    for s in [500, 2000, 7000]:
        print(f"{s:<6} | {results_x[s]:.4f}               | {results_d[s]:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
