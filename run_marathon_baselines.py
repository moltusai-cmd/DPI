import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
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

def get_scheduler(optimizer, total_steps):
    warmup_steps = int(0.02 * total_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def tfixup_init(model, n_layers):
    print(f"Applying Scaled-Init (T-Fixup inspired, L={n_layers})...")
    L = n_layers
    for name, p in model.named_parameters():
        if p.dim() > 1:
            if 'embedding' in name: nn.init.normal_(p, std=1.0)
            elif 'W_o' in name or 'W2' in name: nn.init.zeros_(p)
            elif 'attn' in name or 'mlp' in name:
                std = 0.67 * (L ** (-2/3))
                nn.init.normal_(p, std=std)
            else: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

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

def run_tfixup_marathon():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    
    test_dir = "tests/Marathon_Baselines"
    os.makedirs(test_dir, exist_ok=True)
    
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=320, n_heads=5, d_mlp=1280, n_layers=8, dropout=0.1).to(device)
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=100000)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    print("\n>>> STARTING T-FIXUP RUN (5 Epochs)")
    tfixup_init(model, n_layers=8)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 5
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    global_step = 0
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            if (global_step <= 500 and global_step % 10 == 0) or (global_step > 500 and global_step % 500 == 0):
                elapsed = time.time() - start_time
                val_loss = evaluate(model, val_loader, device)
                lr = optimizer.param_groups[0]['lr']
                print(f"[TFixup] Step {global_step:5d}/{total_steps} | Loss: {loss.item():.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
                history.append({"step": global_step, "train_loss": round(loss.item(), 4), "val_loss": val_loss, "time": round(elapsed, 1)})
                model.train()
                
    with open(f"{test_dir}/marathon_tfixup.json", "w") as f:
        json.dump(history, f, indent=4)
    print("\n>>> T-Fixup Run Complete.")

if __name__ == "__main__":
    run_tfixup_marathon()
