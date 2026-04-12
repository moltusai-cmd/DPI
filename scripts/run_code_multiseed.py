import torch
import torch.nn as nn
import sys
import os
import time
import json
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import math

# Add src to path for core logic
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. REPRODUCIBILITY SETUP ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. CODE DATASET ---
class CodeDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_samples=10000):
        self.seq_len = seq_len
        print(f"  [DATA] Loading CodeSearchNet Python (samples limit: {max_samples})...")
        ds = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
        self.data = []
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            code = ex.get('whole_func_code') or ex.get('func_code_string') or ex.get('code')
            if code:
                self.data.extend(tokenizer.encode(code).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"  [DATA] Loaded {self.num_samples} sequences.")

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

def evaluate(model, loader, device, max_steps=50):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_steps: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
    return round(total_loss / count, 4)

# --- 3. TRAINING ENGINE ---
def run_single_experiment(seed, mode, warmup_pct, device, full_dataset, indices, split, vocab_size, total_steps=7000):
    set_seed(seed)
    print(f"\n[RUN] Seed: {seed:4d} | Mode: {mode:6s} | Warmup: {warmup_pct*100:.1f}%")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(Subset(full_dataset, indices[:split]), batch_size=32, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(full_dataset, indices[split:]), batch_size=32, shuffle=False)
    
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    if mode == "dpi":
        initialize_dpi(model, train_loader, warp_zeta=1.1, spectral_gamma=0.25, use_calibration=True, use_exact_svd=True)
    else:
        xavier_init(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    warmup_steps = int(warmup_pct * total_steps)


    
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    milestones = [500, 2000, 7000]
    
    model.train()
    step = 0
    it = iter(train_loader)
    while step < total_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step in milestones:
            val_loss = evaluate(model, val_loader, device)
            ppl = math.exp(val_loss)
            results[step] = {"loss": val_loss, "ppl": ppl}
            print(f"  > Step {step:4d} | Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
            model.train()
            
    return results

# --- 4. MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 123, 7]
    output_dir = "experiments/Code_Heterogeneity"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Tokenizer
    vocab_file = "data/tokenizers/bpe_tokenizer/vocab.json"
    merges_file = "data/tokenizers/bpe_tokenizer/merges.txt"
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    vocab_size = tokenizer.get_vocab_size()
    
    # 2. Load Dataset
    full_dataset = CodeDataset(tokenizer, max_samples=15000) # Increased for 7000 steps
    indices = list(range(len(full_dataset)))
    split = int(0.9 * len(full_dataset))
    
    all_metrics = {"dpi": [], "xavier": []}
    
    for mode, warmup in [("dpi", 0.0), ("xavier", 0.02)]:
        for seed in seeds:
            res = run_single_experiment(seed, mode, warmup, device, full_dataset, indices, split, vocab_size, total_steps=500)
            all_metrics[mode].append(res)
            
    # --- 5. STATS ---
    print("\n" + "="*75)
    print("🏆 CODE DOMAIN MULTI-SEED ROBUSTNESS REPORT (N=3)")
    print("="*75)
    print(f"{'Step':<6} | {'DPI (Loss)':<15} | {'DPI (PPL)':<12} | {'Xavier (Loss)':<15} | {'Xavier (PPL)'}")
    print("-" * 75)
    
    summary = {}
    for step in [500]:
        d_loss = [m[step]["loss"] for m in all_metrics["dpi"]]
        d_ppl = [m[step]["ppl"] for m in all_metrics["dpi"]]
        x_loss = [m[step]["loss"] for m in all_metrics["xavier"]]
        x_ppl = [m[step]["ppl"] for m in all_metrics["xavier"]]
        
        mu_dl, std_dl = np.mean(d_loss), np.std(d_loss)
        mu_dp, std_dp = np.mean(d_ppl), np.std(d_ppl)
        mu_xl, std_xl = np.mean(x_loss), np.std(x_loss)
        mu_xp, std_xp = np.mean(x_ppl), np.std(x_ppl)
        
        print(f"{step:<6} | {mu_dl:.3f}±{std_dl:.3f} | {mu_dp:.1f}±{std_dp:.1f} | {mu_xl:.3f}±{std_xl:.3f} | {mu_xp:.1f}±{std_xp:.1f}")
        
        summary[step] = {
            "dpi": {"loss": mu_dl, "loss_std": std_dl, "ppl": mu_dp, "ppl_std": std_dp},
            "xavier": {"loss": mu_xl, "loss_std": std_xl, "ppl": mu_xp, "ppl_std": std_xp}
        }

    final_output = {"metadata": {"seeds": seeds, "domain": "python"}, "summary": summary, "raw": all_metrics}
    with open(f"{output_dir}/code_multiseed_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
    print("="*75)

if __name__ == "__main__":
    main()
