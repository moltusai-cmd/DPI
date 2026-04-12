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

def main():
    device = torch.device("cuda")
    cache_file = "checkpoints/wiki_bpe_100000.pt"
    dataset = FastWikiDataset(cache_file)
    train_loader = DataLoader(Subset(dataset, range(1000)), batch_size=32, shuffle=True)
    
    print(f"🚀 Testing DPI v16.2 (Phase-Shift + Zero-Wait Head)")
    torch.manual_seed(42)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    # Measure Step 0 Loss before optimization
    initialize_dpi(model, train_loader, mlp_jitter=0.02)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    it = iter(train_loader)
    x, y = next(it)
    with torch.no_grad():
        logits = model(x.to(device))
        loss_step0 = criterion(logits.view(-1, logits.size(-1)), y.to(device).view(-1))
    
    print(f"\n🔥 STEP 0 VERDICT:")
    print(f"   Validation Loss @ Step 0: {loss_step0.item():.4f}")
    
    # Run 100 steps to see descent
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(1, 101):
        try: x, y = next(it)
        except StopIteration: it = iter(train_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); optimizer.step()
        if step % 20 == 0:
            print(f"   Step {step:3d} | Train Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
