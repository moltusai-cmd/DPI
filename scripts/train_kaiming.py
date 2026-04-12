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

def kaiming_init(model):
    print("Applying Kaiming Uniform Initialization...")
    for name, p in model.named_parameters():
        if p.dim() > 1:
            # We use nonlinearity='linear' for Transformer projections or 'relu' for MLP W1
            if 'mlp.W1' in name:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5), nonlinearity='leaky_relu')
            else:
                nn.init.kaiming_uniform_(p, nonlinearity='linear')
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

def run_kaiming():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8, dropout=0.1).to(device)
    
    kaiming_init(model)
    
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=100000)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(train_loader)
    
    # Same scheduler as the duels
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
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 500 == 0 or step == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"Kaiming | Step {step+1:4d} | Loss: {loss.item():.4f} | Val: {val_loss:.4f}")
            history.append({"step": step + 1, "val_loss": val_loss})
            model.train()
            
    with open("kaiming_baseline.json", "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    run_kaiming()
