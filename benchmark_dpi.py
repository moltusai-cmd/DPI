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

class RobustDataset(Dataset):
    def __init__(self, vocab_size=16384, seq_len=128, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
            text = " ".join(dataset["text"][:1000])
        except Exception:
            text = ("DPI v16.2 Phase-Shift Genomic is the state-of-the-art framework for "
                   "Transformer initialization. " * 500)
        words = text.split()
        tokens = [hash(w) % vocab_size for w in words]
        if len(tokens) < (num_samples + 1) * seq_len:
            tokens = (tokens * ((num_samples * seq_len // len(tokens)) + 2))
        self.tokens = torch.tensor(tokens[:(num_samples + 1) * seq_len])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def get_stable_rank(model, layer_idx, threshold=0.01):
    """Calculates the stable rank with an optional energy threshold."""
    try:
        W = model.layers[layer_idx].attn.W_q.weight.detach()
        # SVD approach for thresholded rank
        U, S, V = torch.linalg.svd(W, full_matrices=False)
        # Filter singular values below threshold relative to max
        S_filtered = S[S > (threshold * S[0])]
        # Stable Rank = ||S||_F^2 / ||S||_2^2
        return (torch.sum(S_filtered**2) / (S[0]**2)).item()
    except Exception:
        return 0.0

def train_model(name, model, loader, val_loader, device, total_steps=1000, warmup_pct=0.02, lr=1e-4, use_scheduler=True):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    if use_scheduler:
        warmup_steps = int(total_steps * warmup_pct)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
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
    parser = argparse.ArgumentParser(description="DPI 8-Test Benchmark (20M Scale)")
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
    print(f"Benchmark: 8 Tests | Model: 20.32M Parameters | Device: {device}")

    configs = [
        ("Xavier Uniform", lambda m: [nn.init.xavier_uniform_(p) for n, p in m.named_parameters() if 'weight' in n and p.dim() >= 2]),
        ("Xavier muP", lambda m: [ (nn.init.xavier_uniform_(p), p.data.mul_(0.1) if any(k in n for k in ['W_q', 'W_k', 'W_v', 'W1']) else None) for n, p in m.named_parameters() if 'weight' in n and p.dim() >= 2]),
        ("DPI v16.2", lambda m: initialize_dpi(m, train_loader, mode="v16.2")),
        ("MuDPI (DPI+muP)", lambda m: [initialize_dpi(m, train_loader, mode="v16.2"), [p.data.mul_(0.1) for n, p in m.named_parameters() if 'weight' in n and any(k in n for k in ['W_q', 'W_k', 'W_v', 'W1'])]])
    ]

    all_results = []
    for init_name, init_fn in configs:
        for sched_type in ["Cosine+Warmup", "Fixed 1e-4"]:
            print(f"Running {init_name} [{sched_type}]...")
            m = copy.deepcopy(model_proto)
            init_fn(m)
            rank = get_stable_rank(m, 3, threshold=0.01)
            use_sched = (sched_type == "Cosine+Warmup")
            warmup = 0.005 if "DPI" in init_name else 0.02
            loss = train_model(init_name, m, train_loader, val_loader, device, total_steps=args.steps, warmup_pct=warmup, use_scheduler=use_sched)
            all_results.append((init_name, sched_type, loss, rank))

    print(f"\n{'='*90}")
    print(f"{'Initialization':<20} | {'Scheduler':<15} | {'Val Loss':<10} | {'Advantage':<10} | {'Rank (0.01)'}")
    print(f"{'-'*90}")
    # Use Xavier Uniform [Cosine+Warmup] as base for advantage
    base_loss = all_results[0][2]
    for name, sched, loss, rank in all_results:
        adv = base_loss - loss
        print(f"{name:<20} | {sched:<15} | {loss:<10.4f} | {adv:<10.4f} | {rank:<10.2f}")
    print(f"{'='*90}")

if __name__ == "__main__":
    main()
