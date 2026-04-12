import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from datasets import load_dataset
import os
import json
import time
from tokenizers import ByteLevelBPETokenizer

class CodeDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_samples=5000):
        self.seq_len = seq_len
        print(f"Loading CodeSearchNet Python (streaming)...")
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        self.data = []
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            code = ex.get('whole_func_code') or ex.get('func_code_string') or ex.get('code')
            if code:
                self.data.extend(tokenizer.encode(code).ids)
        self.num_samples = (len(self.data) - 1) // seq_len

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def xavier_init(model):
    for name, p in model.named_parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
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
    return round(total_loss / min(len(loader), 50), 4)

def run_code_duel(name, init_type):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=320, n_heads=5, d_mlp=1280, n_layers=8, dropout=0.1).to(device)
    
    full_dataset = CodeDataset(tokenizer, seq_len=128, max_samples=5000)
    indices = list(range(len(full_dataset)))
    split = int(0.9 * len(full_dataset))
    train_loader = DataLoader(Subset(full_dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, indices[split:]), batch_size=32, shuffle=False)
    
    if init_type == "dpi":
        print(f"\n>>> [CODE DUEL] DPI_PID14")
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl: yield x.to(device)
        initialize_pid8(model, SL(train_loader), use_whitening=False)
    else:
        print(f"\n>>> [CODE DUEL] Xavier Baseline")
        xavier_init(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    start_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        model.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 100 == 0 or step == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  Step {step+1:4d} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")
            history.append({"step": step + 1, "train_loss": round(loss.item(), 4), "val_loss": val_loss})
            
    return history

if __name__ == "__main__":
    results = {}
    results["dpi"] = run_code_duel("DPI", "dpi")
    torch.cuda.empty_cache(); time.sleep(5)
    results["xavier"] = run_code_duel("Xavier", "xavier")
    
    os.makedirs("tests/Code_Heterogeneity", exist_ok=True)
    with open("tests/Code_Heterogeneity/code_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nCODE HETEROGENEITY TEST COMPLETE.")
