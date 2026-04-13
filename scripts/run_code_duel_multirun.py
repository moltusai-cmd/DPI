import torch
import torch.nn as nn
import sys
import os
import time
import json
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# Performance Boosters
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

class CodeDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_samples=10000):
        self.seq_len = seq_len
        print(f"🚀 Loading CodeSearchNet Python (samples: {max_samples})...")
        # trust_remote_code is removed as per HF update
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
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

def run_session(mode, loader, val_loader, device, seed, vocab_size, total_steps=1000):
    print(f"\n🚀 Starting Run: {mode.upper()} (Seed: {seed})")
    torch.manual_seed(seed)
    
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=8, use_rope=True).to(device)
    
    if mode == "xavier":
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        warmup_steps = 20
    else:
        initialize_dpi(model, loader, mode="v16.2", spectral_gamma=0.25)
        warmup_steps = 0

    try:
        model = torch.compile(model)
    except: pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = {}
    it = iter(loader)
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        
        if step <= warmup_steps:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4 * (step / warmup_steps)

        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step == 1 or step % 200 == 0 or step == total_steps:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                val_it = iter(val_loader)
                for _ in range(20):
                    vx, vy = next(val_it)
                    v_logits = model(vx.to(device))
                    v_loss += criterion(v_logits.view(-1, v_logits.size(-1)), vy.to(device).view(-1)).item()
            avg_v = v_loss / 20
            history[step] = round(avg_v, 4)
            print(f"  [{mode}] Step {step:4d} | Val Loss: {avg_v:.4f}")
            model.train()
            
    return history

def main():
    device = torch.device("cuda")
    vocab_file = "data/tokenizers/bpe_tokenizer/vocab.json"
    merges_file = "data/tokenizers/bpe_tokenizer/merges.txt"
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    vocab_size = tokenizer.get_vocab_size()

    dataset = CodeDataset(tokenizer, max_samples=10000)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    seeds = [42, 43, 44, 45, 46] # PASSAGE À 5 SEEDS
    results = {"xavier": [], "dpi": []}
    
    for seed in seeds:
        results["xavier"].append(run_session("xavier", train_loader, val_loader, device, seed, vocab_size))
        torch.cuda.empty_cache(); time.sleep(2)
        results["dpi"].append(run_session("v16.2", train_loader, val_loader, device, seed, vocab_size))
        torch.cuda.empty_cache(); time.sleep(2)
        
    # Analyse Statistique Finale
    print("\n" + "="*85)
    print(f"📊 CODE DUEL PENTAGON (5 SEEDS) FINAL RESULTS")
    print("="*85)
    print(f"{'Step':<6} | {'Xavier (Mean ± Std)':<25} | {'DPI (Mean ± Std)':<25} | {'Advantage'}")
    print("-" * 85)
    
    for step in [1, 200, 400, 600, 800, 1000]:
        x_vals = [r[step] for r in results["xavier"]]
        d_vals = [r[step] for r in results["dpi"]]
        mx, sx = np.mean(x_vals), np.std(x_vals)
        md, sd = np.mean(d_vals), np.std(d_vals)
        
        x_str = f"{mx:.4f} ± {sx:.4f}"
        d_str = f"{md:.4f} ± {sd:.4f}"
        print(f"{step:<6} | {x_str:<25} | {d_str:<25} | {mx-md:+.4f}")
    
    with open("code_multirun_results_5seeds.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
