import torch
import torch.nn as nn
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import time
import os
import json

class WikiDataset(torch.utils.data.Dataset):
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

def run_duel(mode):
    # Set seeds for absolute bit-perfect reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    
    # 20M Model
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=512, n_heads=8, d_mlp=2048, n_layers=6, dropout=0.1).to(device)
    
    dataset = WikiDataset(tokenizer, seq_len=128, max_lines=20000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"\n>>> [TRIPLE DUEL] Mode: {mode}...")
    
    if mode == "Xavier":
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
    elif mode == "DPI-Full":
        initialize_pid8(model, loader, use_calibration=True)
    elif mode == "DPI-NoCalib":
        initialize_pid8(model, loader, use_calibration=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    start_time = time.time()
    for step, (x, y) in enumerate(loader):
        if step >= 500: break # Extended to 500 steps
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step+1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f}")
            history.append({"step": step+1, "loss": loss.item()})
            
    return history[-1]['loss']

if __name__ == "__main__":
    results = {}
    results["Xavier"] = run_duel("Xavier")
    results["DPI-Full"] = run_duel("DPI-Full")
    results["DPI-NoCalib"] = run_duel("DPI-NoCalib")
    
    print("\n--- FINAL TRIPLE DUEL VERDICT (20M) ---")
    for m, l in results.items():
        print(f"{m:12s}: Loss {l:.4f}")
