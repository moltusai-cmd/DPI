import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, Subset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. GENOMIC DATA GENERATOR ---
def generate_dna_data(length=200000):
    bases = ['A', 'C', 'G', 'T']
    motifs = ['ATG', 'TATA', 'GGCC', 'AAAAAA', 'CCCGGG']
    data = []
    i = 0
    while i < length:
        if random.random() < 0.2: # Insert motif
            m = random.choice(motifs)
            data.extend(list(m))
            i += len(m)
        else:
            data.append(random.choice(bases))
            i += 1
    
    token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '[PAD]': 5}
    return torch.tensor([token_map[b] for b in data[:length]], dtype=torch.long)

class DNADataset(Dataset):
    def __init__(self, tensor, seq_len=64):
        self.seq_len = seq_len
        self.data = tensor
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return (self.data[start : start + self.seq_len],
                self.data[start + 1 : start + self.seq_len + 1])

# --- 2. DUEL ENGINE ---
def run_dna_session(mode, loader, val_loader, device):
    torch.manual_seed(42)
    # Small model for small vocab
    model = PID8Transformer(vocab_size=6, d_model=128, n_heads=4, d_mlp=512, n_layers=6).to(device)
    
    if mode == "xavier":
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        warmup = 100
    else:
        # DPI v15.2
        initialize_dpi(model, loader, use_attention_arch=True, alignment_peak=0.40)
        warmup = 0
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Higher LR for small vocab
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    it = iter(loader)
    for step in range(1000):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                val_it = iter(val_loader)
                for _ in range(10):
                    vx, vy = next(val_it)
                    v_logits = model(vx.to(device))
                    v_loss += criterion(v_logits.view(-1, v_logits.size(-1)), vy.to(device).view(-1)).item()
            avg_v = v_loss / 10
            history.append(avg_v)
            print(f"  [{mode.upper()}] Step {step+1:4d} | Val Loss: {avg_v:.4f}")
            model.train()
            
    return history

def main():
    device = torch.device("cuda")
    print("🧬 Generating Synthetic Genomic Manifold...")
    dna_tensor = generate_dna_data(250000)
    dataset = DNADataset(dna_tensor)
    train_loader = DataLoader(Subset(dataset, range(0, 3000)), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(3000, 3500)), batch_size=32, shuffle=False)
    
    h_xavier = run_dna_session("xavier", train_loader, val_loader, device)
    h_dpi = run_dna_session("dpi", train_loader, val_loader, device)
    
    print("\n" + "="*50)
    print("🧬 GENOMIC UNIVERSALITY REPORT")
    print("="*50)
    print(f"{'Step':<6} | {'Xavier':<12} | {'DPI v15.2':<12} | {'Delta'}")
    print("-" * 50)
    for i, step in enumerate(range(100, 1100, 100)):
        diff = h_dpi[i] - h_xavier[i]
        print(f"{step:<6} | {h_xavier[i]:.4f}      | {h_dpi[i]:.4f}      | {diff:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
