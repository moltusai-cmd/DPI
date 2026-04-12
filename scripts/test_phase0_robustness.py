import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import time
import os
import re

class WikiDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=20000):
        self.seq_len = seq_len
        self.data = []
        with open("wiki.train.raw", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                self.data.extend(tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def run_experiment(name, phase0_lines):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    class SampleLoader:
        def __init__(self, lines): self.lines = lines
        def __iter__(self):
            with open("wiki.train.raw", 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= self.lines: break
                    yield torch.tensor(tokenizer.encode(line).ids).unsqueeze(0).to(device)

    print(f"\n>>> Initializing {name} (Phase 0 lines: {phase0_lines})...")
    initialize_pid8(model, SampleLoader(phase0_lines), use_whitening=False)
    
    dataset = WikiDataset(tokenizer, seq_len=128, max_lines=20000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= 500: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step+1) % 100 == 0 or step == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f}")
            history.append(loss.item())
    
    return history

if __name__ == "__main__":
    res_sparse = run_experiment("DPI-Ultra-Sparse", 100)
    res_std = run_experiment("DPI-Standard", 10000)
    
    print("\n--- FINAL COMPARISON (Step 500) ---")
    print(f"Ultra-Sparse (100 lines) Loss: {res_sparse[-1]:.4f}")
    print(f"Standard (10,000 lines) Loss: {res_std[-1]:.4f}")
    delta = abs(res_sparse[-1] - res_std[-1])
    print(f"Absolute Delta: {delta:.4f}")
    
    if delta < 0.1:
        print("\nCONCLUSION: Phase 0 is IMMUNE to sampling density up to Step 500.")
    else:
        print("\nCONCLUSION: Variance is emerging.")
