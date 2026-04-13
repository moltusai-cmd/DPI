import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import os
import time
import math
import json
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer, count_parameters
from initialize_dpi import initialize_dpi

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
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

def get_scheduler(optimizer, total_steps, warmup_ratio=0.0):
    warmup_steps = int(warmup_ratio * total_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_training(name, init_mode, warmup_ratio):
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = WikiDataset("data/raw/wiki.train.raw", tokenizer)
    
    batch_size = 64
    accumulation_steps = 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if init_mode == "dpi":
        print(f"\n[INIT] {name} - v16.2 Optimized")
        initialize_dpi(model, loader, mode="v16.2", spectral_gamma=0.25)
    else:
        print(f"\n[INIT] {name} - Xavier Uniform")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 2
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    scheduler = get_scheduler(optimizer, total_steps, warmup_ratio=warmup_ratio)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    global_opt_step = 0
    
    print(f"Starting {name} Training ({total_steps} opt steps)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
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
                global_opt_step += 1
                
                if global_opt_step % 10 == 0:
                    print(f"  {name} | Step {global_opt_step}/{total_steps} | Loss: {loss.item():.4f}")
                history.append({"step": global_opt_step, "loss": round(loss.item(), 4)})
                
    return history

if __name__ == "__main__":
    results = {}
    
    # Run DPI (0% Warmup)
    results["dpi"] = run_training("DPI", "dpi", warmup_ratio=0.0)
    torch.cuda.empty_cache(); time.sleep(5)
    
    # Run Xavier (2% Warmup)
    results["xavier"] = run_training("Xavier", "xavier", warmup_ratio=0.02)
    
    with open("duel_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nDUEL COMPLETE. Results saved to duel_results.json")
