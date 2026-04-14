import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mup
import time
import json
import math
import os
from tokenizers import ByteLevelBPETokenizer
from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- DATASET & UTILS ---
class FastArxivDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len=256):
        data_path = "data/raw/arxiv.train.raw"
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read(1000000) # Load enough for grid search
        self.data = tokenizer.encode(text).ids
        self.seq_len = seq_len
    def __len__(self): return len(self.data) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_effective_rank(W, threshold=0.001):
    if W.dim() > 2: W = W.view(-1, W.size(-1))
    s = torch.linalg.svdvals(W.detach().float())
    return (s > threshold * s[0]).sum().item()

def run_trial(name, lr, use_dpi, loader, device, warmup_steps=2000, total_steps=1000):
    print(f"  [Trial] Testing {name} with LR={lr:.1e}...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    cfg = dict(vocab_size=tokenizer.get_vocab_size(), d_model=1024, n_heads=16, n_layers=24, d_mlp=4096, 
               use_rope=True, use_mup_attn=True, use_mup_readout=True, use_rmsnorm=True, use_swiglu=True)
    
    model = PID8Transformer(**cfg).to(device)
    model.gradient_checkpointing = True 
    base_model = PID8Transformer(**{**cfg, 'd_model': 128, 'd_mlp': 512})
    mup.set_base_shapes(model, base_model)
    
    if use_dpi:
        initialize_dpi(model, loader, mode="v16.3")
    else:
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding): mup.init.normal_(m.weight, std=1.0)
            elif isinstance(m, mup.MuReadout): mup.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear): mup.init.normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
            
    optimizer = mup.MuAdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    it = iter(loader)
    status = "SUCCESS"
    max_gn = 0.0
    min_rank = 1024
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        if torch.isnan(loss): status = "FAILED (NaN)"; break
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        max_gn = max(max_gn, gn)
        
        scaler.step(optimizer)
        scaler.update()
        
        if step % 200 == 0:
            rank = get_effective_rank(model.layers[12].attn.W_q.weight)
            min_rank = min(min_rank, rank)
            # Xavier Failure Criteria
            if not use_dpi and (rank < 900 or gn > 5.0):
                status = f"FAILED (Rank={rank}, GN={gn:.2f})"; break
    
    print(f"    Result: {status} | MaxGN: {max_gn:.2f} | MinRank: {min_rank}")
    return {"status": status, "max_gn": round(max_gn, 2), "min_rank": min_rank, "lr": lr}

def main():
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = FastArxivDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    results = {"Xavier": [], "MuDPI": []}
    
    print("\n--- GRID SEARCH: XAVIER-muP ---")
    for lr in [1e-4, 2e-4, 3e-4]:
        results["Xavier"].append(run_trial("Xavier", lr, False, loader, device))
        
    print("\n--- GRID SEARCH: MuDPI-v16.3 ---")
    for lr in [3e-4, 5e-4, 8e-4]:
        results["MuDPI"].append(run_trial("MuDPI", lr, True, loader, device))
        
    with open("results_lrcrit_350m.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
