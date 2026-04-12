import torch
import torch.nn as nn
import sys
import os
import time
import json
from torch.utils.data import DataLoader, Dataset, Subset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

class CodeDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_samples=2000):
        self.seq_len = seq_len
        print(f"🚀 Loading CodeSearchNet Python (samples: {max_samples})...")
        # Use streaming to avoid downloading the whole dataset
        ds = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
        self.data = []
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            code = ex.get('whole_func_code') or ex.get('func_code_string') or ex.get('code')
            if code:
                self.data.extend(tokenizer.encode(code).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"  Dataset Loaded: {self.num_samples} samples of length {seq_len}")

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def evaluate(model, loader, device, max_steps=30):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_steps: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return round(total_loss / min(len(loader), max_steps), 4)

def run_session(mode_name, init_type, loader, val_loader, vocab_size, device, steps=300):
    print(f"\n--- [CODE DUEL] {mode_name} Session ---")
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if init_type == "dpi":
        initialize_dpi(model, loader, warp_zeta=1.1, spectral_gamma=0.25)
    else:
        xavier_init(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    it = iter(loader)
    for step in range(steps):
        model.train()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 50 == 0 or step == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  [{mode_name}] Step {step+1:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            history.append({"step": step + 1, "train_loss": round(loss.item(), 4), "val_loss": val_loss})
            
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using existing tokenizer files
    vocab_file = "data/tokenizers/bpe_tokenizer/vocab.json"
    merges_file = "data/tokenizers/bpe_tokenizer/merges.txt"
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    vocab_size = tokenizer.get_vocab_size()
    
    # 1. Load Code Dataset (Python)
    full_dataset = CodeDataset(tokenizer, max_samples=3000)
    indices = list(range(len(full_dataset)))
    split = int(0.9 * len(full_dataset))
    train_loader = DataLoader(Subset(full_dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, indices[split:]), batch_size=32, shuffle=False)
    
    # 2. RUN DUEL
    results = {}
    results["dpi"] = run_session("DPI-14.1", "dpi", train_loader, val_loader, vocab_size, device)
    torch.cuda.empty_cache(); time.sleep(5)
    results["xavier"] = run_session("Xavier", "xavier", train_loader, val_loader, vocab_size, device)
    
    # 3. SAVE
    output_dir = "experiments/Code_Heterogeneity"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/code_results_verified.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ CODE DUEL COMPLETE. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
