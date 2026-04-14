import torch
import torch.nn as nn
import sys
import os
import time
import math
import json
import mup
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

def run_training_duel(mode, loader, device, total_steps=1000):
    print(f"\n🚀 Starting {mode.upper()} Session (Official Implementation)...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # Common Config
    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=5, n_layers=8, use_rope=True)
    
    if mode == "mup":
        # 1. Official muP Setup
        model = PID8Transformer(**cfg, use_mup=True).to(device)
        # For muP, we need a "base" model to define the scaling reference
        # Here base_model is same as model because we are at 20M scale
        base_model = PID8Transformer(**cfg, use_mup=True) 
        mup.set_base_shapes(model, base_model)
        # Use muP's specific weight initialization
        # (Note: muP weights are initialized based on base_shapes)
        for name, p in model.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                # MuReadout and others handled by set_base_shapes + init
                if 'unembed' in name: nn.init.normal_(p, std=0.02)
                else: nn.init.xavier_uniform_(p)
        
        optimizer = mup.MuAdamW(model.parameters(), lr=1e-4)
        warmup = 100
    elif mode == "xavier":
        model = PID8Transformer(**cfg, use_mup=False).to(device)
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        warmup = 20
    else: # dpi
        model = PID8Transformer(**cfg, use_mup=False).to(device)
        initialize_dpi(model, loader)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        warmup = 0 
    
    criterion = nn.CrossEntropyLoss()
    history = []
    
    it = iter(loader)
    model.train()
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        # Simple linear warmup
        if step <= warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4 * (step / warmup)

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
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = FastArxivDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    res_mup = run_training_duel("mup", loader, device)
    torch.cuda.empty_cache(); time.sleep(1)
    
    res_xav = run_training_duel("xavier", loader, device)
    torch.cuda.empty_cache(); time.sleep(1)
    
    res_dpi = run_training_duel("dpi", loader, device)
    
    print("\n" + "="*80)
    print(f"🏆 THE OFFICIAL BOSS DUEL: muP (Microsoft) vs. Xavier vs. DPI")
    print("="*80)
    print(f"{'Step':<10} | {'muP (Official)':<15} | {'Xavier':<15} | {'DPI (Geometric)':<15} | {'Delta (D-M)'}")
    print("-" * 80)
    
    mup_dict = {h['step']: h['loss'] for h in res_mup}
    xav_dict = {h['step']: h['loss'] for h in res_xav}
    for h in res_dpi:
        s = h['step']
        d_loss = h['loss']
        m_loss = mup_dict.get(s, 0)
        x_loss = xav_dict.get(s, 0)
        print(f"{s:<10} | {m_loss:<15.4f} | {x_loss:<15.4f} | {d_loss:<15.4f} | {d_loss - m_loss:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
