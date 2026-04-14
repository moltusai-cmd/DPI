import torch
import torch.nn as nn
import sys
import os
import math
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

def run_session(name, loader, device, options, total_steps=500):
    print(f"\n🚀 Running Ablation: {name}...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    cfg = dict(
        vocab_size=vocab_size, d_model=320, n_heads=5, n_layers=8, 
        use_rope=True, 
        use_mup_attn=options.get('use_mup_attn', False),
        use_mup_readout=options.get('use_mup_readout', False)
    )
    model = PID8Transformer(**cfg).to(device)
    initialize_dpi(model, loader, mode="v16.2.1")
    
    base_lr = 1e-4
    param_groups = []
    for n, p in model.named_parameters():
        lr = base_lr
        if options.get('lr_embed_10x', False) and "embedding" in n:
            lr = base_lr * 10
        if options.get('lr_readout_01x', False) and "unembed" in n:
            lr = base_lr / 10
        param_groups.append({'params': [p], 'lr': lr})
    
    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = 50
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss()
    it = iter(loader)
    model.train()
    
    last_loss = 0
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
        last_loss = loss.item()
        if step == 1 or step % 100 == 0:
            print(f"  [{name}] Step {step:4d} | Loss: {last_loss:.4f}")
            
    return round(last_loss, 4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    dataset = FastArxivDataset(ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt"))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    results = {}
    
    # 0. Baseline (Gold Standard)
    results['Baseline'] = run_session("Baseline", loader, device, {})
    
    # 1. Attention Scaling (1/d)
    results['Ablation_Attn_1d'] = run_session("Attn_1d", loader, device, {'use_mup_attn': True})
    
    # 2. Readout Logit Scaling (1/d)
    results['Ablation_Logit_1d'] = run_session("Logit_1d", loader, device, {'use_mup_readout': True})
    
    # 3. LR Embedding Boost (10x)
    results['Ablation_LR_Embed'] = run_session("LR_Embed", loader, device, {'lr_embed_10x': True})
    
    # 4. LR Readout Reduction (0.1x)
    results['Ablation_LR_Readout'] = run_session("LR_Readout", loader, device, {'lr_readout_01x': True})
    
    print("\n" + "="*60)
    print(f"🔬 muP FEATURE ABLATION RESULTS (Step 500)")
    print("="*60)
    print(f"{'Configuration':<25} | {'Final Loss':<12} | {'Delta vs Base':<10}")
    print("-" * 60)
    base = results['Baseline']
    for name, loss in results.items():
        delta = base - loss
        print(f"{name:<25} | {loss:<12.4f} | {delta:<10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
