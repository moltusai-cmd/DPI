import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import math
import numpy as np
import random
import sys
import os
import mup

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi
from optimizer import DPISpectralOptimizer

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.inv_vocab = {}
    def encode(self, text):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if h not in self.inv_vocab: self.inv_vocab[h] = word
        return tokens
    def decode(self, token_ids):
        return " ".join([self.inv_vocab.get(tid, f"[{tid}]") for tid in token_ids])

class RobustDataset(Dataset):
    def __init__(self, split="train", target_tokens=1_000_000, noise_prob=0.0):
        self.seq_len = 128
        self.noise_prob = noise_prob
        self.tokenizer = SimpleBPETokenizer(16384)
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
        all_tokens = []
        for entry in dataset:
            all_tokens.extend(self.tokenizer.encode(entry["text"]))
            if len(all_tokens) >= target_tokens: break
        self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len].clone()
        y = self.tokens[start + 1 : start + self.seq_len + 1].clone()
        if self.noise_prob > 0:
            mask = torch.rand(x.shape) < self.noise_prob
            x[mask] = torch.randint(0, 16384, (mask.sum(),))
        return x, y

def calculate_stable_rank(model, threshold=0.01):
    model.eval()
    with torch.no_grad():
        W = model.layers[3].attn.W_q.weight.data
        _, S, _ = torch.svd(W)
        S_filtered = S[S > (threshold * S[0])]
        return (torch.sum(S_filtered**2) / (S[0]**2)).item()

def train_model(name, model, loader, val_loader, device, total_steps=2000, lr=8e-4, sched_type="Cosine", opt_type="AdamW"):
    model.train()
    if opt_type == "DSO":
        optimizer = DPISpectralOptimizer(model.parameters(), lr=lr, weight_decay=0.01, anchor_factor=0.42)
    elif opt_type == "MuAdamW":
        optimizer = mup.MuAdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    warmup_steps = 0
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        if sched_type == "Fixed": return 1.0
        progress = step / total_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    steps = 0
    rank_at_5_5 = None
    steps_at_5_5 = None
    
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
            scheduler.step()
            steps += 1
            
            if steps % 100 == 0:
                model.eval()
                v_loss = 0
                with torch.no_grad():
                    for i, (vx, vy) in enumerate(val_loader):
                        if i > len(val_loader) // 4: break 
                        vx, vy = vx.to(device), vy.to(device)
                        vl = model(vx)
                        v_loss += criterion(vl.view(-1, vl.size(-1)), vy.view(-1)).item()
                current_v_loss = v_loss / (len(val_loader) // 4 + 1)
                if current_v_loss <= 5.5 and rank_at_5_5 is None:
                    rank_at_5_5 = calculate_stable_rank(model)
                    steps_at_5_5 = steps
                    print(f"  [PARITY HIT] {name} reached 5.5 at step {steps} | Rank: {rank_at_5_5:.2f}")
                model.train()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return val_loss / len(val_loader), rank_at_5_5, steps_at_5_5

def generate_inference(model, tokenizer, prompt, device, max_len=30):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    # Simple decoder approximation
    return tokenizer.decode(input_ids[0].cpu().numpy())

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, seq_len = 16384, 128
    
    train_dataset = RobustDataset(split="train", target_tokens=40_000_000, noise_prob=0.0)
    val_dataset = RobustDataset(split="validation", target_tokens=1_000_000, noise_prob=0.0)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=10, d_mlp=1280, n_layers=6, max_len=seq_len, use_mup_attn=True, use_mup_readout=True)
    base_cfg = dict(vocab_size=vocab_size, d_model=64, n_heads=4, d_mlp=256, n_layers=6, max_len=seq_len, use_mup_attn=True, use_mup_readout=True)
    base_model = PID8Transformer(**base_cfg).to(device)
    
    print(f"Giga-Benchmark: (DPI+DSO vs muP) | Seed: {seed} | Parity Target: 5.5 | Device: {device}")
    all_results = []

    for init_name in ["DPI v17.0 + DSO v1.4", "Xavier Uniform", "True muP (MS)", "DPI v17.0 (Static)"]:
        for lr in [1e-4, 8e-4]:
            for sched in ["Cosine", "Fixed"]:
                print(f"Running {init_name} | {sched} @ {lr}...")
                m = PID8Transformer(**cfg).to(device)
                mup.set_base_shapes(m, base_model)
                start_init = time.time()
                if init_name == "Xavier Uniform":
                    for p in m.parameters(): 
                        if p.dim() >= 2: nn.init.xavier_uniform_(p)
                elif init_name == "True muP (MS)":
                    for p in m.parameters():
                        if p.dim() >= 2: mup.init.normal_(p, std=0.02)
                elif "DPI" in init_name:
                    initialize_dpi(m, train_loader, mode="v17.0")
                
                init_time = time.time() - start_init
                rank_before = calculate_stable_rank(m)
                
                start_train = time.time()
                if "DSO" in init_name: opt_type = "DSO"
                elif "muP" in init_name: opt_type = "MuAdamW"
                else: opt_type = "AdamW"
                
                loss, r_5_5, s_5_5 = train_model(init_name, m, train_loader, val_loader, device, lr=lr, sched_type=sched, opt_type=opt_type)
                train_time = time.time() - start_train
                rank_after = calculate_stable_rank(m)

                prompts = ["the story follows", "the game of", "the imperial unit", "in the year", "the system is", "however , the", "according to the", "it is a", "the structure of", "this results in"]
                print(f"  [INFERENCE RESULTS for {init_name}]:")
                for p in prompts:
                    gen = generate_inference(m, train_dataset.tokenizer, p, device)
                    print(f"    - '{p}': {gen}")
                
                all_results.append((init_name, f"{sched} @ {lr}", loss, rank_before, rank_after, r_5_5, s_5_5, train_time))
                print(f"  [RESULT] Loss: {loss:.4f} | Rank Post: {rank_after:.2f} | Rank@5.5: {r_5_5 if r_5_5 else 'N/A'}")
                print(f"{'-'*120}")

    print(f"\n{'='*155}")
    print(f"{'Initialization':<20} | {'Regime':<18} | {'Val Loss':<10} | {'Rank Pre':<10} | {'Rank Post':<10} | {'Rank@5.5':<12} | {'Steps@5.5':<10} | {'Train(s)'}")
    print(f"{'-'*155}")
    for name, sched, loss, r_pre, r_post, r_5_5, s_5_5, t_train in all_results:
        print(f"{name:<20} | {sched:<18} | {loss:<10.4f} | {r_pre:<10.2f} | {r_post:<10.2f} | {f'{r_5_5:.2f}' if r_5_5 else 'N/A':<12} | {f'{s_5_5}' if s_5_5 else 'N/A':<10} | {t_train:<8.1f}")

if __name__ == "__main__":
    main()
