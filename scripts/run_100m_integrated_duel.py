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
    def __init__(self, tokenizer, seq_len=128, max_lines=30000):
        self.seq_len = seq_len
        self.data = []
        file_path = "data/raw/arxiv.train.raw"
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                self.data.extend(tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"  Dataset loaded: {self.num_samples} samples.")
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def run_session(name, loader, device, options, total_steps=2000):
    print(f"\n🚀 Starting 100M Session: {name}...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # 100M PARAMETERS ARCHITECTURE
    cfg = dict(
        vocab_size=vocab_size, d_model=768, n_heads=12, n_layers=12, d_mlp=3072,
        use_rope=True, use_mup_attn=True, use_mup_readout=True
    )
    model = PID8Transformer(**cfg).to(device)
    
    # Official muP Shape Registration
    base_cfg = cfg.copy(); base_cfg['d_model'] = 128; base_cfg['d_mlp'] = 512
    base_model = PID8Transformer(**base_cfg)
    mup.set_base_shapes(model, base_model)
    
    # Initial Weights
    if options.get('use_dpi', False):
        initialize_dpi(model, loader, mode="v16.3.1")
    else:
        # ELITE CANONICAL muP Initialization (Ultra-Fair Baseline)
        print(f"  [muP] Applying Elite Scaling (Emb=1.0, Hidden=Xavier, Readout=0.02)...")
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding):
                mup.init.normal_(m.weight, std=1.0)
            elif isinstance(m, mup.MuReadout):
                mup.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                mup.init.normal_(m.weight, std=1.0 / math.sqrt(fan_in))
            
    # Optimizer - Official muP Transfer (2e-4 for 100M)
    optimizer = mup.MuAdamW(model.parameters(), lr=options['base_lr'])
    
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
        scheduler.step()
        if step % 200 == 0 or step == 1:
            print(f"  [{name}] Step {step:4d} | Loss: {loss.item():.4f}")
            history.append({"step": step, "loss": round(loss.item(), 4)})
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    dataset = FastArxivDataset(ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt"))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    total_steps = 2000
    base_lr = 2e-4
    
    # THE GRAND SCALE DUEL (100M)
    res_fusion = run_session("DPI_MUP_FUSION_100M", loader, device, {'use_dpi': True, 'base_lr': base_lr}, total_steps=total_steps)
    res_mup = run_session("PURE_MUP_OFFICIAL_ELITE_100M", loader, device, {'use_dpi': False, 'base_lr': base_lr}, total_steps=total_steps)
    
    print("\n" + "="*80)
    print(f"🏆 THE GRAND SCALE DUEL: DPI-MUP FUSION VS PURE MUP OFFICIAL (100M)")
    print("="*80)
    print(f"{'Step':<10} | {'Pure muP':<15} | {'DPI-muP Fusion':<15} | {'Delta':<10}")
    print("-" * 80)
    for i in range(len(res_mup)):
        p = res_mup[i]; f = res_fusion[i]
        delta = p['loss'] - f['loss']
        print(f"{p['step']:<10} | {p['loss']:<15.4f} | {f['loss']:<15.4f} | {delta:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
