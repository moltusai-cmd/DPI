import torch
import torch.nn as nn
import sys
import os
import time
import math
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

def run_hybrid_duel(mode, loader, device, total_steps=1000):
    print(f"\n🚀 Starting {mode.upper()} Session...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # Common Config
    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=5, n_layers=8, use_rope=True)
    
    if mode == "mup_stochastic":
        # Standard muP (Microsoft)
        model = PID8Transformer(**cfg, use_mup=True).to(device)
        base_model = PID8Transformer(**cfg, use_mup=True)
        mup.set_base_shapes(model, base_model)
        # Standard Xavier for muP
        for name, p in model.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                if 'unembed' in name: nn.init.normal_(p, std=0.02)
                else: nn.init.xavier_uniform_(p)
        optimizer = mup.MuAdamW(model.parameters(), lr=1e-4)
        warmup = 100 # muP needs warmup for stochastic start
        
    elif mode == "mup_dpi_hybrid":
        # THE HYBRID: muP Mechanics + DPI Geometry
        model = PID8Transformer(**cfg, use_mup=True).to(device)
        base_model = PID8Transformer(**cfg, use_mup=True)
        mup.set_base_shapes(model, base_model)
        
        # CORE STEP: Use DPI to overwrite the weights
        # This keeps muP's attention scaling (1/d) but uses DPI's manifold
        initialize_dpi(model, loader)
        
        optimizer = mup.MuAdamW(model.parameters(), lr=1e-4)
        warmup = 0 # DPI mandate: No Warmup
        
    else: # dpi_standard
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
        
        # Warmup handling
        if mode == "mup_stochastic":
            if step <= warmup:
                for pg in optimizer.param_groups:
                    pg['lr'] = 1e-4 * (step / warmup)
        # Note: Hybrid and DPI use 0% warmup as per DPI v16.2 protocol

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
    
    # 1. muP Official (Stochastic)
    res_mup = run_hybrid_duel("mup_stochastic", loader, device, total_steps=2000)
    torch.cuda.empty_cache(); time.sleep(1)
    
    # 2. DPI Standard (AdamW)
    res_dpi = run_hybrid_duel("dpi_standard", loader, device, total_steps=2000)
    torch.cuda.empty_cache(); time.sleep(1)
    
    # 3. muP + DPI HYBRID
    res_hybrid = run_hybrid_duel("mup_dpi_hybrid", loader, device, total_steps=2000)
    
    print("\n" + "="*90)
    print(f"🏆 THE SINGULARITY DUEL: Official muP vs. DPI vs. muP+DPI HYBRID")
    print("="*90)
    print(f"{'Step':<10} | {'muP (Stoch)':<15} | {'DPI (Std)':<15} | {'muP+DPI (Hybrid)':<20} | {'Advantage'}")
    print("-" * 90)
    
    mup_dict = {h['step']: h['loss'] for h in res_mup}
    dpi_dict = {h['step']: h['loss'] for h in res_dpi}
    for h in res_hybrid:
        s = h['step']
        h_loss = h['loss']
        m_loss = mup_dict.get(s, 0)
        d_loss = dpi_dict.get(s, 0)
        # Advantage of Hybrid over muP alone
        adv = m_loss - h_loss
        print(f"{s:<10} | {m_loss:<15.4f} | {d_loss:<15.4f} | {h_loss:<20.4f} | {adv:+.4f}")
    print("="*90)

if __name__ == "__main__":
    main()
