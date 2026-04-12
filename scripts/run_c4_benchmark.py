import torch
import torch.nn as nn
import sys
import os
import time
import json
import random
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- 1. CONFIGURATION ---
MODEL_CONFIG = {
    "d_model": 1024,
    "n_heads": 16,
    "d_mlp": 4096,
    "n_layers": 24,
    "vocab_size": 16384, # Standard for our BPE
}

TRAIN_CONFIG = {
    "total_steps": 20000,
    "batch_size": 4, # Reduced from 32 to 4 to save VRAM
    "grad_accum": 32, # Effective batch: 128 (4 * 32)
    "lr": 1e-4,      # Standard LR
    "seq_len": 512,
    "weight_decay": 0.01,
    "clip": 1.0,
}


# --- 2. DATASET (STREAMING C4) ---
class C4StreamDataset(IterableDataset):
    def __init__(self, tokenizer, split="train", seq_len=512, max_samples=500000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True, trust_remote_code=True)

    def __iter__(self):
        buffer = []
        count = 0
        for ex in self.dataset:
            if count >= self.max_samples: break
            tokens = self.tokenizer.encode(ex['text']).ids
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                buffer = buffer[self.seq_len:]
                count += 1

# --- 3. LAMBADA EVALUATOR ---
def evaluate_lambada(model, tokenizer, device, num_samples=500):
    """Zero-shot accuracy on LAMBADA (last word prediction)."""

    model.eval()
    print(f"  [EVAL] Running LAMBADA Zero-Shot Probing ({num_samples} samples)...")
    try:
        ds = load_dataset("lambada", split="validation", streaming=True, trust_remote_code=True)
    except:
        # Fallback if lambada is slow to load
        return 0.0
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, ex in enumerate(ds):
            if i >= num_samples: break
            text = ex['text']
            words = text.split()
            if len(words) < 2: continue
            
            context = " ".join(words[:-1])
            target_word = words[-1]
            
            # Encode context
            input_ids = torch.tensor([tokenizer.encode(context).ids], device=device)
            target_ids = tokenizer.encode(" " + target_word).ids
            if not target_ids: continue
            
            logits = model(input_ids) # [1, seq, vocab]
            last_logits = logits[0, -1, :]
            pred_id = torch.argmax(last_logits).item()
            
            if pred_id == target_ids[0]:
                correct += 1
            total += 1
            
    return round(correct / total, 4) if total > 0 else 0.0

# --- 4. REPRODUCIBILITY & INIT ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def evaluate_loss(model, loader, device, max_steps=50):
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

# --- 5. TRAINING SESSION ---
def run_benchmark_session(seed, mode, device, tokenizer, train_loader, val_loader):
    set_seed(seed)
    print(f"\n🚀 Starting C4 SESSION | Seed: {seed} | Mode: {mode}")
    
    model = PID8Transformer(**MODEL_CONFIG).to(device)
    model.gradient_checkpointing = True # Vital for 335M on 16GB VRAM
    
    # 1. INITIALIZATION
    if mode == "dpi":
        # S-DPI: Scaled-DPI for deep architectures (24+ layers)
        res_scale = 1.0 / math.sqrt(2 * MODEL_CONFIG["n_layers"])
        initialize_dpi(
            model, train_loader, 
            warp_zeta=1.1, spectral_gamma=0.25, 
            use_calibration=True, use_exact_svd=True,
            residual_scale=res_scale
        )
        warmup_steps = 100 # Minimal warmup for AdamW stability
    else:
        # Xavier with 2% Warmup (140 steps for 7000, 400 steps for 20000)
        xavier_init(model)
        warmup_steps = int(0.02 * TRAIN_CONFIG["total_steps"])
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"], fused=True)
    scaler = torch.amp.GradScaler('cuda') # For stable FP16
    
    # Cosine Scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, TRAIN_CONFIG["total_steps"] - warmup_steps))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    history = {}
    milestones = [10, 50, 100]
    
    step = 0
    it = iter(train_loader)
    model.train()
    
    t0 = time.time()
    while step < TRAIN_CONFIG["total_steps"]:
        optimizer.zero_grad()
        loss_accum = 0
        
        # Gradient Accumulation
        for _ in range(TRAIN_CONFIG["grad_accum"]):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
                
            x, y = x.to(device), y.to(device)
            # Use autocast BF16 for 100M efficiency on RTX 5080
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / TRAIN_CONFIG["grad_accum"]
            
            # BF16 doesn't strictly need scaler, but keeping structure for robustness
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        
        if step in milestones:
            val_loss = evaluate_loss(model, val_loader, device)
            acc = 0.0
            if step >= 10000:
                acc = evaluate_lambada(model, tokenizer, device)
            
            history[step] = {"loss": val_loss, "acc": acc}
            elapsed = (time.time() - t0) / 60
            print(f"  [{mode}] Step {step:5d} | Val Loss: {val_loss:.4f} | LAMBADA: {acc*100:.2f}% | Time: {elapsed:.1f}m")
            model.train()
            
    return history

# --- 6. MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 123, 7]
    
    # Tokenizer
    vocab_file = "data/tokenizers/bpe_tokenizer/vocab.json"
    merges_file = "data/tokenizers/bpe_tokenizer/merges.txt"
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    
    # Loaders (Streamed)
    train_loader = DataLoader(C4StreamDataset(tokenizer, split="train", seq_len=512), batch_size=TRAIN_CONFIG["batch_size"])
    val_loader = DataLoader(C4StreamDataset(tokenizer, split="validation", seq_len=512), batch_size=TRAIN_CONFIG["batch_size"])
    
    results = {"dpi": [], "xavier": []}
    
    for mode in ["dpi", "xavier"]:
        for seed in seeds:
            res = run_benchmark_session(seed, mode, device, tokenizer, train_loader, val_loader)
            results[mode].append(res)
            
    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_c4_335m.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ C4 BENCHMARK COMPLETE.")

if __name__ == "__main__":
    main()

    main()

