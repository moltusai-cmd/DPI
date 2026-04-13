import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer, count_parameters

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        cache_path = f"wiki_bpe_{max_lines}.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    self.data.extend(tokenizer.encode(line).ids)
            torch.save(self.data, cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_scheduler(optimizer, total_steps, warmup_ratio=0.02):
    warmup_steps = int(warmup_ratio * total_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        (loss / accumulation_steps).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Step {i:5d}/{len(loader)} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    return total_loss / len(loader)

def main():
    device = torch.device("cuda")
    
    # 1. Architecture 20M
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    print(f"Model Parameters: {count_parameters(model)/1e6:.2f}M")

    # 2. Xavier Initialization
    print("Applying Xavier Uniform Initialization...")
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

    # 3. Data
    tokenizer_path = "data/tokenizers/bpe_tokenizer"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = WikiDataset("data/raw/wiki.train.raw", tokenizer)
    
    batch_size = 64
    accumulation_steps = 4 # Total effective batch size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Optimizer & Scheduler
    epochs = 2
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = get_scheduler(optimizer, total_steps, warmup_ratio=0.02)
    criterion = nn.CrossEntropyLoss()

    # 5. Training
    print(f"Starting Xavier Training: {epochs} epochs, {total_steps} opt steps...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, scheduler, accumulation_steps)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 6. Save
    torch.save(model.state_dict(), "model_xavier_20m.pt")
    print("\nModel saved to model_xavier_20m.pt")

if __name__ == "__main__":
    main()
