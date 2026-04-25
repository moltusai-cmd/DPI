import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import copy
import numpy as np
import random

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi
from initialize_dpi_legacy import initialize_dpi as initialize_dpi_legacy
from optimizer import SpectreMuon
from muon import SingleDeviceMuonWithAuxAdam

def get_effective_rank(W, threshold=0.01): # Changement ici
    if W.dim() > 2: W = W.view(-1, W.size(-1))
    # On passe en float32 pour la précision de la SVD
    s = torch.linalg.svdvals(W.detach().float())
    # On compte les valeurs singulières significatives
    rank = (s > threshold * s[0]).sum().item()
    return f"{rank}/{min(W.size(0), W.size(1))}"

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
    def encode(self, text, target_count=None):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if target_count and len(tokens) >= target_count: break
        return tokens

class RobustDataset(Dataset):
    def __init__(self, split="train", vocab_size=16384, seq_len=128, target_tokens=1_000_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        cache_path = f"results/tokens_cache_{split}_{target_tokens}.pt"
        if os.path.exists(cache_path):
            self.tokens = torch.load(cache_path)
        else:
            print(f"📦 Tokenizing {split} split...")
            tokenizer = SimpleBPETokenizer(vocab_size)
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
            all_tokens = []
            for entry in dataset:
                all_tokens.extend(tokenizer.encode(entry["text"]))
                if len(all_tokens) >= target_tokens: break
            self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
            os.makedirs("results", exist_ok=True)
            torch.save(self.tokens, cache_path)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def setup_muon_optimizer(model, base_lr=1e-4):
    hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "unembed" not in n]
    embed_params = [p for n, p in model.named_parameters() if ("embed" in n or "unembed" in n) and p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    
    adam_groups = [
        dict(params=embed_params, lr=base_lr),
        dict(params=scalar_params, lr=base_lr)
    ]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    # To ensure a perfectly fair 1:1 comparison against AdamW/DSO, 
    # we force Muon to use the exact same global learning rate (base_lr)
    muon_group = dict(params=hidden_matrix_params, lr=base_lr, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    return SingleDeviceMuonWithAuxAdam(param_groups)

def train_model(name, model, loader, val_loader, device, opt_type="SpectreMuon", total_steps=2000, lr=1e-4):
    model.train()
    if opt_type == "SpectreMuon":
        optimizer = SpectreMuon(model.parameters(), lr=lr, weight_decay=0.01, anchor_factor=0.5)
    elif opt_type == "Muon":
        optimizer = setup_muon_optimizer(model, base_lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    # Cosine scheduler
    warmup_steps = 100
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
            scheduler.step()
            steps += 1
            if steps % 100 == 0:
                # Calculate effective rank for a mid layer's W_q
                rank = get_effective_rank(model.layers[len(model.layers)//2].attn.W_q.weight)
                print(f"  [{name}] Step {steps:4d} | Loss: {loss.item():.4f} | ERank W_q: {rank}")
            if torch.isnan(loss):
                print(f"  [{name}] 💥 CRASHED at step {steps}")
                return 99.99
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return val_loss / len(val_loader)

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, seq_len = 16384, 128
    
    # 20 Million tokens to ensure 2000 steps (32 * 128 = 4096 tokens/step -> ~8.2M tokens needed)
    # plus the tokens consumed by initialize_dpi.
    # This prevents the model from seeing the same data multiple times (no epochs looping).
    train_dataset = RobustDataset(split="train", target_tokens=20_000_000)
    val_dataset = RobustDataset(split="validation", target_tokens=100_000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 100M params model architecture
    cfg = dict(vocab_size=vocab_size, d_model=768, n_heads=12, d_mlp=3072, n_layers=12, max_len=seq_len, use_mup_attn=False, use_mup_readout=False)
    
    print(f"⚔️ ULTIMATE DUEL 100M (2000 steps, Cosine LR 1e-e3): DPI+SpectreMuon vs DPI (Multiple versions)+Muon vs Xavier+Muon")
    print(f"{'='*80}")

    lr = 1e-3
    results = {}

    # 1. DPI + SpectreMuon (Baseline reference for spectral clipping)
    print("\n🚀 Running DPI v17.0 + SpectreMuon...")
    torch.manual_seed(seed)
    m_dpi_dso = PID8Transformer(**cfg).to(device)
    initialize_dpi(m_dpi_dso, train_loader, mode="v17.0")
    results["DPI_v17.0_SpectreMuon"] = train_model("DPI v17.0 + SpectreMuon", m_dpi_dso, train_loader, val_loader, device, opt_type="SpectreMuon", lr=lr, total_steps=2000)

    # 2. DPI + Muon (Ablation sur les versions)
    dpi_versions = ["v14", "v15", "v16.2", "v16.3", "v17.0"]
    for v in dpi_versions:
        print(f"\n🚀 Running DPI {v} + Muon...")
        torch.manual_seed(seed)
        m_dpi_muon = PID8Transformer(**cfg).to(device)
        if v == "v14":
            initialize_dpi_legacy(m_dpi_muon, train_loader, use_attention_arch=False)
        elif v == "v15":
            initialize_dpi_legacy(m_dpi_muon, train_loader, use_attention_arch=True)
        else:
            initialize_dpi(m_dpi_muon, train_loader, mode=v)
        results[f"DPI_{v}_Muon"] = train_model(f"DPI {v} + Muon", m_dpi_muon, train_loader, val_loader, device, opt_type="Muon", lr=lr, total_steps=2000)

    # 3. Xavier + Muon
    print("\n🚀 Running Xavier + Muon...")
    torch.manual_seed(seed)
    m_xavier_muon = PID8Transformer(**cfg).to(device)
    for p in m_xavier_muon.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)
    results["Xavier_Muon"] = train_model("Xavier+Muon", m_xavier_muon, train_loader, val_loader, device, opt_type="Muon", lr=lr, total_steps=2000)

    print(f"\n{'='*60}")
    print(f"{'Configuration':<30} | {'Val Loss':<10}")
    print(f"{'-'*60}")
    for k, v in results.items():
        print(f"{k:<30} | {v:<10.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
()

