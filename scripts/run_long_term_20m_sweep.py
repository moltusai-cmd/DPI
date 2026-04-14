import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import json
import mup
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer, count_parameters
from initialize_dpi import initialize_dpi

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=1800000, name="train"):
        self.seq_len = seq_len
        cache_path = f"wiki_bpe_{max_lines}_{name}.pt"
        if os.path.exists(cache_path):
            print(f"Loading {name} cache: {cache_path}")
            self.data = torch.load(cache_path)
        else:
            self.data = []
            print(f"Tokenizing {name} ({max_lines} lines)...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    self.data.extend(tokenizer.encode(line).ids)
            torch.save(self.data, cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"{name.capitalize()} Dataset: {self.num_samples} samples")

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def run_experiment(lr, name, epochs=3, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[STARTING] {name} - LR: {lr}")
    
    tokenizer_path = "data/tokenizers/bpe_tokenizer"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    
    train_dataset = WikiDataset("data/raw/wiki.train.raw", tokenizer, max_lines=1800000, name="train")
    val_dataset = WikiDataset("data/raw/wiki.valid.raw", tokenizer, max_lines=100000, name="val")
    
    batch_size = 64
    num_workers = 4 # High speed loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    # Target Architecture (20M)
    cfg = dict(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8)
    model = PID8Transformer(**cfg).to(device)
    
    # Base Model for muP
    base_cfg = cfg.copy(); base_cfg['d_model'] = 128; base_cfg['d_mlp'] = 512
    base_model = PID8Transformer(**base_cfg)
    mup.set_base_shapes(model, base_model)
    
    # DPI Initialization
    print(f"Initializing DPI v16.2...")
    initialize_dpi(model, train_loader, mode="v16.2", spectral_gamma=0.25)
    
    # MAX OPTIMIZATION: Compile the model (PyTorch 2.0+)
    print("Compiling model (torch.compile)...")
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # For mixed precision if BF16 is not available
    
    history = []
    global_step = 0
    total_opt_steps = math.ceil(len(train_loader) / accumulation_steps) * epochs
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # MIXED PRECISION (BF16 prefered for stability)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {name} | Epoch {epoch+1} | Step {global_step}/{total_opt_steps} | Loss: {loss.item()*accumulation_steps:.4f} | {elapsed/60:.1f}m")
                
                if global_step % 10 == 0:
                    history.append({"step": global_step, "loss": round(loss.item()*accumulation_steps, 4)})
        
        # Validation at end of epoch
        val_loss = validate(model, val_loader, criterion, device)
        print(f"  >>> {name} | Epoch {epoch+1} | Validation Loss: {val_loss:.4f}")
        history[-1]["val_loss"] = round(val_loss, 4)
                    
    return history

if __name__ == "__main__":
    lrs = [1e-4, 1e-5, 5e-6]
    results = {}
    
    for lr in lrs:
        name = f"DPI_LR_{lr}"
        results[name] = run_experiment(lr, name, epochs=3)
        torch.cuda.empty_cache()
        time.sleep(10) # Cooling down GPU
    
    os.makedirs("results", exist_ok=True)
    with open("results/long_term_20m_sweep.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nSWEEP COMPLETE. Results saved to results/long_term_20m_sweep.json")
