import torch
import torch.nn as nn
import bitsandbytes as bnb
import sys
import os
import time
import math
import json
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer

class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=40000):
        self.seq_len = seq_len
        self.data = []
        file_path = "data/raw/arxiv.train.raw"
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                self.data.extend(tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"  Dataset Loaded: {self.num_samples} samples.")
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def apply_linear_depth_scaling(state_dict, n_layers=40):
    # Aggressive Scaling: 1/L (0.025 for 40 layers)
    scale_factor = 1.0 / n_layers 
    print(f"  Applying Linear Depth Scaling (Factor: {scale_factor:.4f})")
    for k in state_dict.keys():
        if "attn.W_o.weight" in k or "mlp.W2.weight" in k:
            state_dict[k] = state_dict[k] * scale_factor
    return state_dict

def get_scheduler(optimizer, warmup_steps=100):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_8b_strong_scale_accum():
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    
    print(f"\n🚀 [TITAN 8B] Starting L-DPI Experiment (Batch Size: 8 via Accumulation)")
    print(f"Target LR: 1e-4 | Warmup: 100 updates")
    
    # 1. Build Model
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0, use_rope=False)
    
    # 2. Load Weights
    ckpt_path = "checkpoints/dpi_8b_bf16.pt"
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = apply_linear_depth_scaling(state_dict, n_layers=40)
    
    # 3. Quantize
    def replace_with_4bit(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                new_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=torch.bfloat16)
                setattr(model, name, new_layer)
            else: replace_with_4bit(module)
    
    print("  Quantizing to 4-bit (NF4)...")
    replace_with_4bit(model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.gradient_checkpointing = True
    
    # 4. Setup
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=40000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    scheduler = get_scheduler(optimizer, warmup_steps=100)
    criterion = nn.CrossEntropyLoss()
    
    accumulation_steps = 8
    total_updates = 1000
    
    print(f">>> RUNNING {total_updates} UPDATES (Total Samples: {total_updates * accumulation_steps})...")
    history = []
    start_time = time.time()
    model.train()
    
    loader_iter = iter(loader)
    
    for update in range(1, total_updates + 1):
        optimizer.zero_grad()
        update_loss = 0.0
        
        for _ in range(accumulation_steps):
            try: x, y = next(loader_iter)
            except StopIteration: loader_iter = iter(loader); x, y = next(loader_iter)
            
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / accumulation_steps
            
            loss.backward()
            update_loss += loss.item()
        
        # Calculate GN (after accumulation)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if update % 20 == 0 or update == 1:
            print(f"  Update {update:4d}/{total_updates} | Loss: {update_loss:.4f} | GN: {total_norm:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            history.append({"update": update, "loss": round(update_loss, 4), "gn": round(total_norm, 4)})

    with open("results_8b_strong_scale_bs8.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n✅ Experiment Complete. Results saved to results_8b_strong_scale_bs8.json")

if __name__ == "__main__":
    run_8b_strong_scale_accum()
