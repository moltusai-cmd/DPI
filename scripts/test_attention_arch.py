import torch
import torch.nn as nn
import sys
import os
import time
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
        return (torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long),
                torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long))

# --- 2. TRAINING ENGINE ---
def run_training(mode, device, loader, val_loader, total_steps=500):
    torch.manual_seed(42)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if mode == "gold":
        print(f"\n🚀 Mode: DPI Gold Standard (Linear QK Alignment)")
        initialize_dpi(model, loader, use_attention_arch=False, mlp_jitter=0.02)
    else:
        print(f"\n🚀 Mode: DPI Genomic 15.2 (Attention Alignment Arch)")
        initialize_dpi(model, loader, use_attention_arch=True, mlp_jitter=0.02)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {}
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
        step += 1
        
        if step % 100 == 0:
            model.eval()
            total_v_loss = 0
            with torch.no_grad():
                val_it = iter(val_loader)
                for _ in range(20):
                    vx, vy = next(val_it)
                    v_logits = model(vx.to(device))
                    total_v_loss += criterion(v_logits.view(-1, v_logits.size(-1)), vy.to(device).view(-1)).item()
            val_loss = total_v_loss / 20
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
    
    res_gold = run_training("gold", device, train_loader, val_loader)
    torch.cuda.empty_cache(); time.sleep(2)
    res_genomic = run_training("genomic", device, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("🏆 ATTENTION ARCH DUEL: GOLD vs 15.2")
    print("="*60)
    print(f"{'Step':<6} | {'Gold (Linear)':<20} | {'Genomic 15.2 (Arch)'}")
    print("-" * 60)
    for s in [100, 300, 500]:
        diff = res_genomic[s] - res_gold[s]
        sign = "+" if diff > 0 else ""
        print(f"{s:<6} | {res_gold[s]:.4f}               | {res_genomic[s]:.4f} ({sign}{diff:.4f})")
    print("="*60)

if __name__ == "__main__":
    main()
