import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import copy
import json
import argparse

# Add src to path for imports
sys.path.append('src')
try:
    from model import PID8Transformer, count_parameters
    from initialize_dpi import initialize_dpi
except ImportError:
    print("❌ Error: Could not find 'src/model.py' or 'src/initialize_dpi.py'.")
    print("Please run this script from the project root.")
    sys.exit(1)

class SimpleBPETokenizer:
    """A minimal, deterministic BPE-like tokenizer for the benchmark."""
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.byte_encoder = self.bytes_to_unicode()
        
    def bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    def encode(self, text, target_count=None):
        # Deterministic hashing of characters to tokens for a BPE-like feel 
        # without the overhead of training a full vocabulary.
        # This ensures punctuation and words are mapped consistently.
        tokens = []
        for word in text.split():
            # Simple deterministic word-to-id mapping
            h = 0
            for char in word:
                h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if target_count and len(tokens) >= target_count:
                break
        return tokens

class RobustDataset(Dataset):
    def __init__(self, vocab_size=16384, seq_len=128, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        tokenizer = SimpleBPETokenizer(vocab_size)
        
        try:
            from datasets import load_dataset
            print(f"📦 Loading WikiText-103 (Target: 40M tokens)...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            
            print("🔢 Encoding tokens...")
            all_tokens = []
            target_tokens = 40_000_000
            
            for entry in dataset:
                line_tokens = tokenizer.encode(entry["text"])
                all_tokens.extend(line_tokens)
                if len(all_tokens) >= target_tokens:
                    break
            
            self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens)/1e6:.1f}M tokens")
            
        except Exception as e:
            print(f"⚠️ Falling back to Deep Relativity Corpus (Reason: {e})")
            text = ("General Relativity describes gravity as a geometric property of spacetime. " * 5000)
            all_tokens = tokenizer.encode(text)
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        if len(self.tokens) < (num_samples + 1) * seq_len:
            repeat_factor = ((num_samples * seq_len) // len(self.tokens)) + 1
            self.tokens = self.tokens.repeat(repeat_factor)

    def __getitem__(self, idx):
        # Use random offset for each fetch to maximize data variety if num_samples is large
        start = idx * self.seq_len
        if start + self.seq_len + 1 > len(self.tokens):
            start = 0
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def get_stable_rank(model, layer_idx, threshold=0.01):
    try:
        W = model.layers[layer_idx].attn.W_q.weight.detach()
        U, S, V = torch.linalg.svd(W, full_matrices=False)
        S_filtered = S[S > (threshold * S[0])]
        return (torch.sum(S_filtered**2) / (S[0]**2)).item()
    except Exception:
        return 0.0

def get_stable_decay_lambda(total_steps, stable_ratio=0.8):
    stable_steps = int(total_steps * stable_ratio)
    def lr_lambda(step):
        if step < stable_steps: return 1.0
        progress = (step - stable_steps) / (total_steps - stable_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

def train_model(name, model, loader, val_loader, device, total_steps=1000, warmup_pct=0.02, lr=1e-4, sched_type="Cosine"):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    if "Cosine" in sched_type:
        warmup_steps = int(total_steps * warmup_pct)
        def lr_lambda(step):
            if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif "Stable-Decay" in sched_type:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_stable_decay_lambda(total_steps))
    else: # Fixed
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    steps = 0
    while steps < total_steps:
        for x, y in loader:
            if steps >= total_steps: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            steps += 1
            
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return val_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description="DPI 16-Test Benchmark (20M Scale)")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, seq_len = 16384, 128
    train_dataset = RobustDataset(vocab_size=vocab_size, num_samples=args.steps * 16)
    val_dataset = RobustDataset(vocab_size=vocab_size, num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model_args = dict(vocab_size=vocab_size, d_model=320, n_heads=10, d_mlp=1280, n_layers=6, max_len=seq_len)
    model_proto = PID8Transformer(**model_args).to(device)
    print(f"Giga-Benchmark: 16 Tests | Model: 20.32M Parameters | Device: {device}")

    inits = [
        ("Xavier Uniform", lambda m: [nn.init.xavier_uniform_(p) for n, p in m.named_parameters() if 'weight' in n and p.dim() >= 2]),
        ("Xavier muP", lambda m: [ (nn.init.xavier_uniform_(p), p.data.mul_(0.1) if any(k in n for k in ['W_q', 'W_k', 'W_v', 'W1']) else None) for n, p in m.named_parameters() if 'weight' in n and p.dim() >= 2]),
        ("DPI v16.2", lambda m: initialize_dpi(m, train_loader, mode="v16.2")),
        ("MuDPI v16.3", lambda m: initialize_dpi(m, train_loader, mode="v16.3"))
    ]

    all_results = []
    for init_name, init_fn in inits:
        for base_lr in [1e-4, 8e-4]:
            # Each init gets two schedulers at each LR
            if "MuDPI" in init_name:
                scheds = ["Stable-Decay", "Fixed"]
            else:
                scheds = ["Cosine+Warmup", "Fixed"]
                
            for sched_type in scheds:
                display_sched = f"{sched_type} @ {base_lr}"
                print(f"Running {init_name} [{display_sched}]...")
                m = copy.deepcopy(model_proto)
                init_fn(m)
                rank = get_stable_rank(m, 3, threshold=0.01)
                warmup = 0.005 if "DPI" in init_name else 0.02
                loss = train_model(init_name, m, train_loader, val_loader, device, total_steps=args.steps, warmup_pct=warmup, lr=base_lr, sched_type=sched_type)
                all_results.append((init_name, display_sched, loss, rank))

    print(f"\n{'='*100}")
    print(f"{'Initialization':<20} | {'Scheduler Regime':<20} | {'Val Loss':<10} | {'Advantage':<10} | {'Rank'}")
    print(f"{'-'*100}")
    base_loss = all_results[0][2] # Xavier [Cosine+Warmup @ 1e-4]
    for name, sched, loss, rank in all_results:
        adv = base_loss - loss
        print(f"{name:<20} | {sched:<20} | {loss:<10.4f} | {adv:<10.4f} | {rank:<10.2f}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
