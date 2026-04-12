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
        return (torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long),
                torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long))

def evaluate(model, loader, device, max_steps=30):
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

# --- 2. SWEEP ENGINE ---
def run_alignment_point(peak_val, loader, val_loader, device, vocab_size, steps=500):
    print(f"\n📈 Testing Alignment Peak = {peak_val:.2f}")
    torch.manual_seed(42)
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    # Init with specific peak
    initialize_dpi(model, loader, use_exact_svd=True, mlp_jitter=0.02, use_attention_arch=True, alignment_peak=peak_val)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    it = iter(loader)
    for step in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
    final_val_loss = evaluate(model, val_loader, device)
    print(f"   Done. Final Val Loss: {final_val_loss:.4f}")
    return final_val_loss

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    peaks = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for p in peaks:
        val_loss = run_alignment_point(p, train_loader, val_loader, device, vocab_size=16384)
        results[p] = val_loss
        torch.cuda.empty_cache()
        
    print("\n" + "="*40)
    print("🏆 ALIGNMENT PEAK SWEEP REPORT")
    print("="*40)
    print(f"{'Peak':<10} | {'Val Loss':<12}")
    print("-" * 25)
    for p in peaks:
        print(f"{p:<10.2f} | {results[p]:.4f}")
    print("="*40)
    
    os.makedirs("results", exist_ok=True)
    with open("results/alignment_peak_sweep.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
