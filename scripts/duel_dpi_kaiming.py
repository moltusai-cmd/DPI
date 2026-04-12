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
        self.tokenizer = tokenizer
        cache_path = f"wiki_bpe_{max_lines}.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    self.data.extend(self.tokenizer.encode(line).ids)
            torch.save(self.data, cache_path)
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

def kaiming_init(model):
    print("Applying PyTorch Default Kaiming (Leaky ReLU a=sqrt(5))...")
    for name, p in model.named_parameters():
        if p.dim() > 1:
            # Emulating standard nn.Linear initialization (PyTorch Default)
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
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

def run_training(mode_name, init_type):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    test_dir = "tests/DPI_vs_Kaiming_5Epoch"
    os.makedirs(test_dir, exist_ok=True)
    
    # Correction 1: Dynamic vocab_size
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=8, dropout=0.1).to(device)
    
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=100000)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    if init_type == "dpi":
        print(f"\n>>> [INIT] DPI (PID-14 Light)")
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl: yield x.to(device)
        initialize_pid8(model, SL(train_loader), use_whitening=False)
    else:
        print(f"\n>>> [INIT] Kaiming Baseline")
        kaiming_init(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 5
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    global_step = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            
            # Correction 3: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            if (global_step <= 500 and global_step % 10 == 0) or (global_step > 500 and global_step % 100 == 0):
                elapsed = time.time() - start_time
                entry = {"step": global_step, "train_loss": round(loss.item(), 4), "time": round(elapsed, 1)}
                if global_step % 500 == 0 or global_step == 1:
                    val_loss = evaluate(model, val_loader, device)
                    entry["val_loss"] = val_loss
                    lr = optimizer.param_groups[0]['lr']
                    print(f"[{mode_name}] Step {global_step:5d}/{total_steps} | Loss: {loss.item():.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
                    model.train()
                history.append(entry)
                
    with open(f"{test_dir}/duel_{mode_name.lower()}.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f">>> {mode_name} Run Complete.")

if __name__ == "__main__":
    run_training("DPI_PID14", "dpi")
    torch.cuda.empty_cache(); time.sleep(5)
    run_training("Kaiming", "kaiming")
    print("\nKAIMING DUEL COMPLETE.")
