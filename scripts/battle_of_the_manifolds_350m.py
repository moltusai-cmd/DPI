import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mup
import time
import json
import math
from tokenizers import ByteLevelBPETokenizer
from model import PID8Transformer
from initialize_dpi import initialize_dpi

# --- DATASET (Full ArXiv) ---
class FastArxivDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len=256):
        data_path = "data/raw/arxiv.train.raw"
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read() 
        self.data = tokenizer.encode(text).ids
        self.seq_len = seq_len
    def __len__(self): return len(self.data) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_effective_rank(W, threshold=0.001):
    if W.dim() > 2: W = W.view(-1, W.size(-1))
    s = torch.linalg.svdvals(W.detach().float())
    return (s > threshold * s[0]).sum().item()

def generate_sample(model, tokenizer, device, prompt, max_new_tokens=50):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    print(f"\n  [Generation] Prompt: {prompt}")
    generated = tokens
    for _ in range(max_new_tokens):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    text = tokenizer.decode(generated)
    print(f"  [Result]: {text}\n")
    return text

def run_session(name, loader, device, options, total_steps=10000):
    print(f"\n🚀 Starting {name} (350M Class)...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    cfg = dict(vocab_size=tokenizer.get_vocab_size(), d_model=1024, n_heads=16, n_layers=24, d_mlp=4096,
               use_rope=True, use_mup_attn=True, use_mup_readout=True, use_rmsnorm=True, use_swiglu=True)
    
    model = PID8Transformer(**cfg).to(device)
    model.gradient_checkpointing = True 
    base_model = PID8Transformer(**{**cfg, 'd_model': 128, 'd_mlp': 512})
    mup.set_base_shapes(model, base_model)
    
    if options.get('use_dpi', False):
        initialize_dpi(model, loader, mode="v16.3")
    else:
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding): mup.init.normal_(m.weight, std=1.0)
            elif isinstance(m, mup.MuReadout): mup.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear): mup.init.normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
            
    optimizer = mup.MuAdamW(model.parameters(), lr=options['lr'])
    scaler = torch.amp.GradScaler('cuda')
    
    warmup_steps = options.get('warmup', 0)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine Decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss()
    history = []
    it = iter(loader)
    
    for step in range(1, total_steps + 1):
        model.train()
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if step in [1, 1000, 5000, 10000] or step % 1000 == 0:
            rank = get_effective_rank(model.layers[12].attn.W_q.weight)
            print(f"  [{name}] Step {step:5d} | Loss: {loss.item():.4f} | GN: {gn:.2f} | Rank: {rank}")
            history.append({"step": step, "loss": loss.item(), "rank": rank, "gn": gn})
            
    # Final Generation
    prompt = "The derivation of the Einstein field equations starts from"
    generate_sample(model, tokenizer, device, prompt)
    
    # Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{name.lower()}_final.pt")
    
    return history

def main():
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = FastArxivDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    total_steps = 10000
    
    # CHALLENGER: MuDPI-v16.3 (LR=8e-4, 0 W)
    res_dpi = run_session("MuDPI_v16_3", loader, device, 
                         {'use_dpi': True, 'lr': 8e-4, 'warmup': 0}, 
                         total_steps=total_steps)
    
    # BASELINE: Xavier-muP (LR=2e-4, 2k W)
    res_xavier = run_session("Xavier_muP", loader, device, 
                            {'use_dpi': False, 'lr': 2e-4, 'warmup': 2000}, 
                            total_steps=total_steps)
    
    results = {"MuDPI": res_dpi, "Xavier": res_xavier}
    with open("results_battle_manifolds_350m.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
