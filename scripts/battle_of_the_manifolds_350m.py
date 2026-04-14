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

# --- DATASET (Full ArXiv with Train/Val Split) ---
class FastArxivDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=256):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def evaluate(model, val_loader, device, criterion, num_batches=50):
    model.eval()
    total_loss = 0
    batches_seen = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_batches: break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            batches_seen += 1
    return total_loss / max(1, batches_seen)

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

def get_stable_decay_lambda(total_steps, stable_ratio=0.8):
    stable_steps = int(total_steps * stable_ratio)
    decay_steps = total_steps - stable_steps
    def lr_lambda(current_step):
        if current_step <= stable_steps: return 1.0
        decay_progress = float(current_step - stable_steps) / float(max(1, decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return lr_lambda

def run_session(name, train_loader, val_loader, device, options, total_steps=10000):
    print(f"\n🚀 Starting {name} (100M Class)...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    cfg = dict(vocab_size=tokenizer.get_vocab_size(), d_model=768, n_heads=12, n_layers=12, d_mlp=2048,
               use_rope=True, use_mup_attn=True, use_mup_readout=True, use_rmsnorm=True, use_swiglu=True)
    
    model = PID8Transformer(**cfg).to(device)
    model.gradient_checkpointing = True 
    base_model = PID8Transformer(**{**cfg, 'd_model': 128, 'd_mlp': 512})
    mup.set_base_shapes(model, base_model)
    
    start_init = time.time()
    if options.get('use_dpi', False):
        initialize_dpi(model, train_loader, mode="v16.3")
    else:
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding): mup.init.normal_(m.weight, std=1.0)
            elif isinstance(m, mup.MuReadout): mup.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear): mup.init.normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
    init_time = time.time() - start_init
    
    optimizer = mup.MuAdamW(model.parameters(), lr=options['lr'])
    scaler = torch.amp.GradScaler('cuda')
    
    # SCHEDULER SELECTION
    if options.get('use_dpi', False):
        # MuDPI: Stable-Decay (Pied au plancher)
        lr_lambda = get_stable_decay_lambda(total_steps, stable_ratio=0.8)
    else:
        # Xavier: Standard Linear Warmup + Cosine Decay
        warmup_steps = options.get('warmup', 2000)
        def lr_lambda(current_step):
            if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    history = []
    it = iter(train_loader)
    
    start_train = time.time()
    train_loss_acc = 0.0
    
    for step in range(1, total_steps + 1):
        model.train()
        try: x, y = next(it)
        except StopIteration: it = iter(train_loader); x, y = next(it)
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
        
        train_loss_acc += loss.item()
        
        if step in [1, 1000, 5000, 10000] or step % 1000 == 0:
            val_loss = evaluate(model, val_loader, device, criterion)
            avg_train_loss = train_loss_acc / (step if step < 1000 else 1000)
            if step % 1000 == 0: train_loss_acc = 0.0 # Reset for local average
            
            elapsed = time.time() - start_train
            mid_layer = model.layers[6].attn.W_q.weight
            rank = get_effective_rank(mid_layer)
            print(f"  [{name}] Step {step:5d} | ValL: {val_loss:.4f} | TrL: {avg_train_loss:.4f} | GN: {gn:.2f} | Rank: {rank} | Time: {elapsed:.1f}s")
            history.append({"step": step, "val_loss": val_loss, "train_loss": avg_train_loss, "rank": rank, "gn": gn, "time_s": elapsed})
            
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
    
    # Load and Split Data (95% Train, 5% Val)
    data_path = "data/raw/arxiv.train.raw"
    with open(data_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    all_tokens = tokenizer.encode(full_text).ids
    
    split_idx = int(len(all_tokens) * 0.95)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    train_dataset = FastArxivDataset(train_tokens)
    val_dataset = FastArxivDataset(val_tokens)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    total_steps = 10000
    
    # CHALLENGER: MuDPI-v16.3 (LR=8e-4, 0 W)
    res_dpi = run_session("MuDPI_v16_3", train_loader, val_loader, device, 
                         {'use_dpi': True, 'lr': 8e-4, 'warmup': 0}, 
                         total_steps=total_steps)
    
    # BASELINE: Xavier-muP (LR=2e-4, 2k W)
    res_xavier = run_session("Xavier_muP", train_loader, val_loader, device, 
                            {'use_dpi': False, 'lr': 2e-4, 'warmup': 2000}, 
                            total_steps=total_steps)
    
    results = {"MuDPI": res_dpi, "Xavier": res_xavier}
    with open("results_battle_manifolds_350m.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
