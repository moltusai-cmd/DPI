import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import math
import time
import json
import os

class WikiDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=100000):
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

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 50: break # Evaluation on 50 batches for speed
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    model.train()
    return round(total_loss / 50, 4)

def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_60m_duel(mode):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=512, n_heads=8, d_mlp=2048, n_layers=14, dropout=0.1).to(device)
    
    full_dataset = WikiDataset(tokenizer, seq_len=128, max_lines=100000)
    indices = list(range(len(full_dataset)))
    split = int(0.9 * len(full_dataset))
    train_loader = DataLoader(Subset(full_dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, indices[split:]), batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    if mode == "DPI-Pure":
        print(f"\n>>> [60M] Initializing DPI (0% Warmup - Pure Velocity)...")
        class SimpleLoader:
            def __iter__(self):
                for x, y in train_loader: yield x.to(device)
        initialize_pid8(model, SimpleLoader(), use_whitening=False)
        warmup_steps = 0
    else:
        print(f"\n>>> [60M] Initializing Xavier (10% Warmup - Industrial Prudence)...")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        warmup_steps = int(0.1 * len(train_loader))

    scheduler = get_warmup_scheduler(optimizer, warmup_steps)
    
    history = []
    model.train()
    start_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (step+1) % 200 == 0 or step == 0:
            val_loss = evaluate(model, val_loader, device)
            avg_t = (time.time()-start_time)/(step+1)
            print(f"  Step {step+1:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            history.append({"step": step+1, "train_loss": round(loss.item(), 4), "val_loss": val_loss})
            
    return history

if __name__ == "__main__":
    results = {}
    results["dpi_pure"] = run_60m_duel("DPI-Pure")
    results["xavier_warmup"] = run_60m_duel("Xavier-Warmup")
    
    os.makedirs("tests/Asymmetric_Duel_60M", exist_ok=True)
    with open("tests/Asymmetric_Duel_60M/results_val.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n>>> 60M ASYMMETRIC DUEL (WITH VAL LOSS) COMPLETE.")
