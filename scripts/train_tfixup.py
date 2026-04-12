import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from model import PID8Transformer
import os
import json
import time
import math
from tokenizers import ByteLevelBPETokenizer

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        cache_path = f"wiki_bpe_{max_lines}.pt"
        self.data = torch.load(cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def tfixup_init(model, n_layers):
    """Simplified T-Fixup scaling for a standard Transformer with LayerNorm."""
    print(f"Applying T-Fixup Style Scaling (L={n_layers})...")
    L = n_layers
    for name, p in model.named_parameters():
        if p.dim() > 1:
            if 'embedding' in name:
                nn.init.normal_(p, std=1.0) # T-Fixup starts with unit var for embeddings
            elif 'W_o' in name or 'W2' in name:
                # T-Fixup Core: Initialize output projections to zero 
                # to make residual blocks act as identity at start.
                nn.init.zeros_(p)
            elif 'attn' in name or 'mlp' in name:
                # T-Fixup scaling factor: 0.67 * L^(-2/3)
                std = 0.67 * (L ** (-2/3))
                nn.init.normal_(p, std=std)
            else:
                nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 50: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return round(total_loss / 50, 4)

def run_tfixup():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    n_layers = 8
    
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=n_layers, dropout=0.1).to(device)
    
    tfixup_init(model, n_layers)
    
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=100000)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(train_loader)
    
    def lr_lambda(current_step):
        warmup = int(0.02 * total_steps)
        if current_step < warmup: return float(current_step) / float(max(1, warmup))
        else:
            progress = float(current_step - warmup) / float(max(1, total_steps - warmup))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 500 == 0 or step == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"T-Fixup | Step {step+1:4d} | Loss: {loss.item():.4f} | Val: {val_loss:.4f}")
            history.append({"step": step + 1, "val_loss": val_loss})
            model.train()
            
    with open("tfixup_baseline.json", "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    run_tfixup()
