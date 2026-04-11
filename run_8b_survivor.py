import torch
import torch.nn as nn
import bitsandbytes as bnb
from model import PID8Transformer, count_parameters
from tokenizers import ByteLevelBPETokenizer
import os
import json
import time
import math

class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=50000):
        self.seq_len = seq_len
        self.data = []
        with open("arxiv.train.raw", 'r', encoding='utf-8') as f:
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

def apply_hard_scaling_to_dpi(state_dict, factor=0.05):
    print(f"Applying HARD SCALING (factor={factor}) to DPI residual projections...")
    for k in state_dict.keys():
        if "attn.W_o.weight" in k or "mlp.W2.weight" in k:
            state_dict[k] = state_dict[k] * factor
    return state_dict

def run_8b_hard_scaled_accumulator():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    print(f"\n>>> [TITAN 8B HARD-SCALED DPI] Factor=0.05 | No Warmup | LR=1e-4")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0)
    
    state_dict = torch.load("dpi_8b_bf16.pt", map_location='cpu', weights_only=True)
    state_dict = apply_hard_scaling_to_dpi(state_dict, factor=0.05)
    
    def convert_to_4bit(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                new_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=torch.bfloat16)
                setattr(model, name, new_layer)
            else: replace_with_4bit(module)
    
    replace_with_4bit(model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.gradient_checkpointing = True
    
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=50000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    accum_steps = 32
    print(f">>> RUNNING 1000 UPDATES (8B Hard-Scaled DPI | Virtual BS={accum_steps})...")
    history = []
    start_time = time.time()
    model.train()
    
    running_loss = 0.0
    optimizer.zero_grad()
    
    for step, (x, y) in enumerate(loader):
        if step >= 1000 * accum_steps: break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / accum_steps
        loss.backward()
        running_loss += loss.item() * accum_steps
        
        if (step + 1) % accum_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            update_idx = (step + 1) // accum_steps
            if update_idx % 10 == 0 or update_idx == 1:
                print(f"  Update {update_idx:4d}/1000 | Loss: {running_loss/accum_steps:.4f} | GN: {total_norm:.4f} | T/update: {time.time()-start_time:.1f}s")
                history.append({"step": update_idx, "loss": round(running_loss/accum_steps, 4), "gn": round(total_norm, 4)})
            running_loss = 0.0

if __name__ == "__main__":
    run_8b_hard_scaled_accumulator()
