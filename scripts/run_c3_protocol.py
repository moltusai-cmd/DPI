import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mup
import math
import time
import json
from tokenizers import ByteLevelBPETokenizer
from model import PID8Transformer
from initialize_dpi import initialize_dpi
from datasets import load_dataset

# --- DATASETS ---

class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=256):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

class CodeDataset(Dataset):
    def __init__(self, tokenizer, seq_len=256, max_tokens=200000000):
        self.seq_len = seq_len
        print(f"Loading CodeSearchNet Python (Massive Transfer Test: 200M tokens)...")
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        self.data = []
        token_count = 0
        for i, ex in enumerate(ds):
            code = ex.get('whole_func_code') or ex.get('func_code_string') or ex.get('code', "")
            ids = tokenizer.encode(code).ids
            self.data.extend(ids)
            token_count += len(ids)
            if token_count >= max_tokens: break
            if i % 10000 == 0: print(f"  Captured {token_count} tokens...")
        print(f"Loaded {token_count} code tokens (Single-Pass Ready).")
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_effective_rank(W, threshold=0.001):
    if W.dim() > 2: W = W.view(-1, W.size(-1))
    s = torch.linalg.svdvals(W.detach().float())
    return (s > threshold * s[0]).sum().item()

def evaluate(model, loader, device, criterion, num_batches=50):
    model.eval()
    total_loss = 0
    batches_seen = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches: break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            batches_seen += 1
    return total_loss / max(1, batches_seen)

# --- RUN SESSION ---

def run_session_c3(name, train_loader, val_loader, dpi_loader, device, options, total_steps=10000):
    print(f"\n🚀 Starting C3 Protocol: {name} (Iso-Params Duel)...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    
    cfg = dict(vocab_size=tokenizer.get_vocab_size(), d_model=768, n_heads=12, n_layers=12, d_mlp=2048,
               use_rope=True, use_mup_attn=True, use_mup_readout=True, use_rmsnorm=True, use_swiglu=True)
    
    model = PID8Transformer(**cfg).to(device)
    model.gradient_checkpointing = True 
    base_model = PID8Transformer(**{**cfg, 'd_model': 128, 'd_mlp': 512})
    mup.set_base_shapes(model, base_model)
    
    # INITIALIZATION
    if options.get('use_dpi', False):
        # Phase 0 Seeding is ALWAYS on ArXiv (dpi_loader)
        print(f"  [Phase 0] Seeding on ArXiv (Domain Transfer Test)...")
        initialize_dpi(model, dpi_loader, mode="v16.3")
    else:
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding): mup.init.normal_(m.weight, std=1.0)
            elif isinstance(m, mup.MuReadout): mup.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear): mup.init.normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
            
    optimizer = mup.MuAdamW(model.parameters(), lr=options['lr'])
    scaler = torch.amp.GradScaler('cuda')
    
    # ISO-PARAMS: 2k Warmup + Cosine Decay
    warmup_steps = 2000
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    history = []
    it = iter(train_loader)
    
    # Metrics tracking
    threshold_5_5_reached = False
    slope_tracking = [] # (step, loss)
    
    for step in range(1, total_steps + 1):
        model.train()
        try: x, y = next(it)
        except StopIteration: it = iter(train_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # AUDIT: Gap de départ (Step 1)
        if step == 1:
            val_loss = evaluate(model, val_loader, device, criterion)
            print(f"  [AUDIT: Gap] Step 1 Val Loss: {val_loss:.4f}")
            history.append({"step": 1, "val_loss": val_loss, "type": "gap_audit"})

        # AUDIT: Pente (Iso-Loss 5.5)
        if not threshold_5_5_reached and loss.item() <= 5.5:
            threshold_5_5_reached = True
            print(f"  [AUDIT: Slope] Threshold 5.5 reached at step {step}. Starting 100-step slope tracking.")
            
        if threshold_5_5_reached and len(slope_tracking) < 100:
            slope_tracking.append(loss.item())
            if len(slope_tracking) == 100:
                slope = (slope_tracking[0] - slope_tracking[-1]) / 100.0
                print(f"  [AUDIT: Slope] Measured dL/dt over 100 steps: {slope:.6f} per step.")
                history.append({"step": step, "slope": slope, "type": "slope_audit"})

        # AUDIT: Intégrité Géométrique (Every 500 steps)
        if step % 500 == 0:
            val_loss = evaluate(model, val_loader, device, criterion)
            rank_q = get_effective_rank(model.layers[6].attn.W_q.weight)
            rank_w1 = get_effective_rank(model.layers[6].mlp.W1.weight)
            print(f"  [AUDIT: Rank] Step {step:5d} | ValL: {val_loss:.4f} | Rank Q: {rank_q} | Rank W1: {rank_w1}")
            history.append({"step": step, "val_loss": val_loss, "rank_q": rank_q, "rank_w1": rank_w1, "gn": gn})

    return history

def main():
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    
    # 1. Prepare ArXiv (for Phase 0 Seeding ONLY)
    data_path = "data/raw/arxiv.train.raw"
    with open(data_path, 'r', encoding='utf-8') as f:
        arxiv_text = f.read(1000000) # Small sample for Phase 0 seeding
    arxiv_tokens = tokenizer.encode(arxiv_text).ids
    arxiv_loader = DataLoader(ArxivDataset(arxiv_tokens), batch_size=64, shuffle=False)
    
    # 2. Prepare Code Dataset (Massive Transfer Test: 200M tokens)
    code_dataset = CodeDataset(tokenizer, max_tokens=200000000) 
    split_idx = int(len(code_dataset.data) * 0.95)
    train_tokens = code_dataset.data[:split_idx]
    val_tokens = code_dataset.data[split_idx:]
    
    train_loader = DataLoader(ArxivDataset(train_tokens), batch_size=64, shuffle=True)
    val_loader = DataLoader(ArxivDataset(val_tokens), batch_size=64, shuffle=False)
    
    total_steps = 10000
    lr_max = 4e-4 # Xavier's limit
    
    # --- DUEL C3 ---
    
    # DPI (Seeded on ArXiv, Trained on Code)
    res_dpi = run_session_c3("MuDPI_C3_Transfer", train_loader, val_loader, arxiv_loader, device, 
                            {'use_dpi': True, 'lr': lr_max}, 
                            total_steps=total_steps)
    
    # XAVIER (Stochastic, Trained on Code)
    res_xavier = run_session_c3("Xavier_C3_Baseline", train_loader, val_loader, arxiv_loader, device, 
                               {'use_dpi': False, 'lr': lr_max}, 
                               total_steps=total_steps)
    
    results = {"MuDPI_C3": res_dpi, "Xavier_C3": res_xavier}
    with open("results_c3_protocol.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
