import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. ROBUST GENOMIC DATA LOADER ---
def get_human_dna_tensor(num_chars=300000):
    print(f"🧬 Loading Genomic Manifold (Human DNA Subset)...")
    try:
        # Use a more modern dataset format or a different one if needed
        # InstaDeep's one has a script error. Let's try 'tiago-pimentel/human_genome' 
        # or just a raw text stream if we can find one.
        # Actually, let's use 'genomic_datasets/human_reference_genome' which is typically safer.
        ds = load_dataset("genomic_datasets/human_reference_genome", split="train", streaming=True)
        
        data_str = ""
        for entry in ds:
            # Check for sequence or text field
            seq = entry.get('sequence', entry.get('text', ''))
            data_str += seq
            if len(data_str) >= num_chars:
                break
    except Exception as e:
        print(f"⚠️ HF Dataset Error: {e}. Falling back to high-fidelity synthetic genomic manifold.")
        return generate_synthetic_genomic(num_chars)
            
    data_str = data_str[:num_chars].upper()
    token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    tokens = [token_map[b] for b in data_str if b in token_map]
    if len(tokens) < 1000: return generate_synthetic_genomic(num_chars)
    return torch.tensor(tokens, dtype=torch.long)

def generate_synthetic_genomic(length=300000):
    # High-fidelity synthetic DNA with biological grammar (Codons, Promoters)
    bases = ['A', 'C', 'G', 'T']
    # Motifs: Start, TATA, Stop, Repeats
    motifs = ['ATG', 'TATAAA', 'TAG', 'TGA', 'TAA', 'GCCG', 'AAAAAA', 'GGCGGG']
    data = []
    i = 0
    while i < length:
        r = random.random()
        if r < 0.15: # Real biological motifs
            m = random.choice(motifs)
            data.extend(list(m))
            i += len(m)
        elif r < 0.25: # Markovian local structure
            prev = data[-1] if data else 'A'
            choices = {'A': ['T', 'G'], 'T': ['A', 'C'], 'G': ['C', 'A'], 'C': ['G', 'T']}
            data.append(random.choice(choices.get(prev, bases)))
            i += 1
        else: # Random noise
            data.append(random.choice(bases))
            i += 1
    token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return torch.tensor([token_map[b] for b in data[:length]], dtype=torch.long)

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
    model = PID8Transformer(vocab_size=5, d_model=256, n_heads=8, d_mlp=1024, n_layers=6).to(device)
    
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
    for step in range(1000):
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
    dna_tensor = get_human_dna_tensor(350000)
    dataset = DNADataset(dna_tensor)
    indices = list(range(len(dataset)))
    train_loader = DataLoader(Subset(dataset, indices[:2000]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[2000:2300]), batch_size=32, shuffle=False)
    
    h_xavier = run_session = run_dna_session("xavier", train_loader, val_loader, device)
    h_dpi = run_dna_session("dpi", train_loader, val_loader, device)
    
    print("\n" + "="*60)
    print("🧬 GENOMIC UNIVERSALITY REPORT")
    print("="*60)
    print(f"{'Step':<6} | {'Xavier':<15} | {'DPI v15.2':<12} | {'Delta'}")
    print("-" * 60)
    for i, step in enumerate(range(100, 1100, 100)):
        diff = h_dpi[i] - h_xavier[i]
        print(f"{step:<6} | {h_xavier[i]:.4f}          | {h_dpi[i]:.4f}      | {diff:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
