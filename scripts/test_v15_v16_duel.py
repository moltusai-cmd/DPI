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
    print(f"\n🚀 Starting Session: DPI {mode.upper()} (Target: {total_steps} steps)")
    torch.manual_seed(42)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    # Init specific mode
    initialize_dpi(model, loader, mode=mode, mlp_jitter=0.02)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = {}
    it = iter(loader)
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); optimizer.step()
        
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
    
    res_v15 = run_session("v15", train_loader, val_loader, device, total_steps=2000)
    torch.cuda.empty_cache(); time.sleep(2)
    res_v16 = run_session("v16", train_loader, val_loader, device, total_steps=2000)
    
    print("\n" + "="*50)
    print("🏆 LONG-TERM DUEL (2000 steps): v15.2 vs v16.0")
    print("="*50)
    print(f"{'Step':<6} | {'v15.2 (Hyper)':<15} | {'v16.0 (Phase)'} | {'Delta'}")
    print("-" * 50)
    for s in [500, 1000, 1500, 2000]:
        diff = res_v16[s] - res_v15[s]
        print(f"{s:<6} | {res_v15[s]:.4f}          | {res_v16[s]:.4f}          | {diff:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
