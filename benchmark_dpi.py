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
import mup
import numpy as np
import random

# Add src to path for imports
sys.path.append('src')
from model import PID8Transformer, count_parameters
from initialize_dpi import initialize_dpi
from optimizer import DPISpectralOptimizer

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.inv_vocab = {} # Mapping from hash to a sample word
    def encode(self, text, target_count=None):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if h not in self.inv_vocab: self.inv_vocab[h] = word # Keep a representative
            if target_count and len(tokens) >= target_count: break
        return tokens

def generate_inference(model, tokenizer, prompt, device, max_len=15):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids[:, -128:])
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    return " ".join([tokenizer.inv_vocab.get(t, f"[{t}]") for t in generated])

class RobustDataset(Dataset):
    def __init__(self, split="train", vocab_size=16384, seq_len=128, target_tokens=1_000_000, noise_prob=0.0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.noise_prob = noise_prob
        self.tokenizer = SimpleBPETokenizer(vocab_size)
        cache_path = f"results/tokens_cache_{split}_{target_tokens}.pt"
        if os.path.exists(cache_path):
            self.tokens = torch.load(cache_path)
            # Comprehensive scan to populate inv_vocab for readable inference
            # We don't have the raw text here, but we can't easily reverse the hash.
            # FIX: We'll re-run a small part of the tokenizer on the first few entries of WikiText
            try:
                from datasets import load_dataset
                sample_data = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
                count = 0
                for entry in sample_data:
                    self.tokenizer.encode(entry["text"])
                    count += 1
                    if count > 500: break # Scan 500 lines to fill the dictionary
            except: pass
        else:
            print(f"📦 Tokenizing {split} split...")
            try:
                from datasets import load_dataset
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
                all_tokens = []
                for entry in dataset:
                    all_tokens.extend(self.tokenizer.encode(entry["text"]))
                    if len(all_tokens) >= target_tokens: break
                self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
                os.makedirs("results", exist_ok=True)
                torch.save(self.tokens, cache_path)
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Could not load WikiText-103 split '{split}'. Reason: {e}")
                sys.exit(1)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len].clone()
        y = self.tokens[start + 1 : start + self.seq_len + 1].clone()
        if self.noise_prob > 0:
            mask = torch.rand(x.shape) < self.noise_prob
            x[mask] = torch.randint(0, self.vocab_size, (mask.sum(),), dtype=torch.long)
        return x, y

def calculate_stable_rank(model, threshold=0.01):
    with torch.no_grad():
        W = model.layers[3].attn.W_q.weight.data
        _, S, _ = torch.svd(W)
        S_filtered = S[S > (threshold * S[0])]
        return (torch.sum(S_filtered**2) / (S[0]**2)).item()

def train_model(name, model, loader, val_loader, device, total_steps=2000, lr=1e-3, sched_type="Cosine", opt_type="AdamW"):
    model.train()
    if opt_type == "DSO":
        optimizer = DPISpectralOptimizer(model.parameters(), lr=lr, weight_decay=0.01, anchor_factor=0.42)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    warmup_steps = 0
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = step / total_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    steps = 0
    rank_at_5_4 = None
    steps_at_5_4 = None
    
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
                if current_v_loss <= 5.4 and rank_at_5_4 is None:
                    rank_at_5_4 = calculate_stable_rank(model)
                    steps_at_5_4 = steps
                    print(f"  [PARITY HIT] {name} reached 5.4 at step {steps} | Rank: {rank_at_5_4:.2f}")
                model.train()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return val_loss / len(val_loader), rank_at_5_4, steps_at_5_4

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, seq_len = 16384, 128
    
    noise_level = 0.0
    train_dataset = RobustDataset(split="train", target_tokens=40_000_000, noise_prob=noise_level)
    val_dataset = RobustDataset(split="validation", target_tokens=1_000_000, noise_prob=0.0)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=10, d_mlp=1280, n_layers=6, max_len=seq_len, use_mup_attn=True, use_mup_readout=True)
    base_cfg = dict(vocab_size=vocab_size, d_model=64, n_heads=4, d_mlp=256, n_layers=6, max_len=seq_len, use_mup_attn=True, use_mup_readout=True)
    base_model = PID8Transformer(**base_cfg).to(device)
    
    print(f"Giga-Benchmark: (DPI+DSO vs muP) | Seed: {seed} | Parity Target: 5.4 | Device: {device}")
    all_results = []

    for init_name in ["DPI v17.0 + DSO v1.4", "Xavier Uniform", "True muP (MS)", "DPI v17.0 (Static)"]:
        for lr in [8e-4]:
            for sched in ["Cosine"]:
                print(f"Running {init_name}...")
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
                opt_type = "DSO" if "DSO" in init_name else "AdamW"
                loss, r_54, s_54 = train_model(init_name, m, train_loader, val_loader, device, lr=lr, sched_type=sched, opt_type=opt_type)
                train_time = time.time() - start_train
                rank_after = calculate_stable_rank(m)

                prompts = ["the story follows", "the game of", "the imperial unit", "in the year", "the system is", "however , the", "according to the", "it is a", "the structure of", "this results in"]
                print(f"  [INFERENCE RESULTS for {init_name}]:")
                for p in prompts:
                    gen = generate_inference(m, train_dataset.tokenizer, p, device)
                    print(f"    - '{p}': {gen}")
                
                all_results.append((init_name, f"{sched} @ {lr}", loss, rank_before, rank_after, r_54, s_54, train_time))
                print(f"  [RESULT] Loss: {loss:.4f} | Rank Post: {rank_after:.2f} | Rank@5.4: {r_54 if r_54 else 'N/A'}")
                print(f"{'-'*120}")

    print(f"\n{'='*155}")
    print(f"{'Initialization':<20} | {'Regime':<18} | {'Val Loss':<10} | {'Rank Pre':<10} | {'Rank Post':<10} | {'Rank@5.4':<12} | {'Steps@5.4':<10} | {'Train(s)'}")
    print(f"{'-'*155}")
    for name, sched, loss, r_pre, r_post, r_54, s_54, t_train in all_results:
        print(f"{name:<20} | {sched:<18} | {loss:<10.4f} | {r_pre:<10.2f} | {r_post:<10.2f} | {f'{r_54:.2f}' if r_54 else 'N/A':<12} | {f'{s_54}' if s_54 else 'N/A':<10} | {t_train:<8.1f}")
    print(f"{'='*155}")

if __name__ == "__main__":
    main()
