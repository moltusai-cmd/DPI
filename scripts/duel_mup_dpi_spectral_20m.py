import torch
import torch.nn as nn
import sys
import os
import math
import mup
import json
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer
from initialize_dpi import initialize_dpi

class FastArxivDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=10000):
        self.seq_len = seq_len
        self.data = []
        file_path = "data/raw/arxiv.train.raw"
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                self.data.extend(tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_spectral_optimizer(model, spectral_map, base_lr=1e-4):
    """
    S-muP: Official muP optimizer with DPI spectral modulation.
    """
    param_groups = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # Identify layer index
        layer_idx = None
        parts = name.split('.')
        if len(parts) > 1 and parts[1].isdigit():
            layer_idx = int(parts[1])
            
        lr = base_lr
        # Apply DPI spectral correction (S-muP)
        if layer_idx is not None and layer_idx in spectral_map:
            # Scale LR based on spectral density gamma
            multiplier = math.sqrt(spectral_map[layer_idx] / 0.25)
            lr *= multiplier
            
        param_groups.append({'params': [p], 'lr': lr})
        
    # muP.MuAdamW will handle the width-based scaling automatically
    # provided that set_base_shapes was called.
    return mup.MuAdamW(param_groups)

def run_spectral_hybrid(loader, device, total_steps=1000):
    print(f"\n🚀 Starting OFFICIAL S-muP (Spectral Hybrid) Session...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # Target Model
    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=5, n_layers=8, use_rope=True, use_mup=True)
    model = PID8Transformer(**cfg).to(device)
    
    # Base Model (smaller width for muP shape inference)
    base_cfg = cfg.copy()
    base_cfg['d_model'] = 160
    base_model = PID8Transformer(**base_cfg) # Stays on CPU
    
    # CRITICAL: Link models for muP
    mup.set_base_shapes(model, base_model)
    
    # 1. DPI Initialization
    spectral_map = initialize_dpi(model, loader)
    
    # 2. Official muP Optimizer with Spectral groups
    optimizer = get_spectral_optimizer(model, spectral_map, base_lr=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    history = []
    it = iter(loader)
    model.train()
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0 or step == 1:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f}")
            history.append({"step": step, "loss": round(loss.item(), 4)})
            
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = FastArxivDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    res_spectral = run_spectral_hybrid(loader, device, total_steps=1000)
    
    print("\n" + "="*60)
    print(f"🏆 OFFICIAL S-muP RESULT")
    print("="*60)
    for h in res_spectral:
        print(f"Step {h['step']:<10} | Loss: {h['loss']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
