import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np
import math
import gzip
import urllib.request
from torch.utils.data import DataLoader, Dataset, Subset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. REAL GENOMIC DATA STREAMER (UCSC) ---
def get_real_human_dna(num_chars=500000):
    print(f"🧬 Streaming Real Human Chromosome 22 (hg38)...")
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
    
    try:
        response = urllib.request.urlopen(url)
        with gzip.GzipFile(fileobj=response) as f:
            data_str = ""
            # Skip the FASTA header
            f.readline()
            
            while len(data_str) < num_chars:
                line = f.readline().decode('ascii').strip().upper()
                if not line: break
                # Only keep ACGT (skip N and other masks)
                filtered = "".join([c for c in line if c in "ACGT"])
                data_str += filtered
                
        print(f"✅ Loaded {len(data_str)} base pairs of real DNA.")
        token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        tokens = [token_map[b] for b in data_str[:num_chars]]
        return torch.tensor(tokens, dtype=torch.long)
    except Exception as e:
        print(f"❌ Error streaming DNA: {e}")
        sys.exit(1)

class DNADataset(Dataset):
    def __init__(self, tensor, seq_len=128):
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
    # 20M-scale equivalent for Vocab=4
    model = PID8Transformer(vocab_size=4, d_model=256, n_heads=8, d_mlp=1024, n_layers=6).to(device)
    
    if mode == "xavier":
        print(f"\n🚀 Mode: XAVIER")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        warmup = 100
    else:
        print(f"\n🚀 Mode: DPI v15.2")
        initialize_dpi(model, loader, use_attention_arch=True, alignment_peak=0.40)
        warmup = 0
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    it = iter(loader)
    for step in range(2000):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); optimizer.step()
        
        if (step + 1) % 100 == 0:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                val_it = iter(val_loader)
                for _ in range(20):
                    try: vx, vy = next(val_it)
                    except StopIteration: val_it = iter(val_loader); vx, vy = next(val_it)
                    v_logits = model(vx.to(device))
                    v_loss += criterion(v_logits.view(-1, v_logits.size(-1)), vy.to(device).view(-1)).item()
            avg_v = v_loss / 20
            history.append(avg_v)
            print(f"  [{mode.upper()}] Step {step+1:4d} | Val Loss: {avg_v:.4f}")
            model.train()
    return history

def main():
    device = torch.device("cuda")
    dna_tensor = get_real_human_dna(400000)
    dataset = DNADataset(dna_tensor)
    indices = list(range(len(dataset)))
    train_loader = DataLoader(Subset(dataset, indices[:2500]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[2500:2800]), batch_size=32, shuffle=False)
    
    h_xavier = run_dna_session("xavier", train_loader, val_loader, device)
    h_dpi = run_dna_session("dpi", train_loader, val_loader, device)
    
    print("\n" + "="*65)
    print("🧬 THE CHROMOSOME 22 DUEL: REAL HUMAN GENOME")
    print("="*65)
    print(f"{'Step':<6} | {'Xavier Baseline':<18} | {'DPI v15.2':<15} | {'Delta'}")
    print("-" * 65)
    for i, step in enumerate(range(200, 2200, 200)):
        diff = h_dpi[i] - h_xavier[i]
        sign = "+" if diff > 0 else ""
        print(f"{step:<6} | {h_xavier[i]:.4f}             | {h_dpi[i]:.4f}          | {sign}{diff:.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
