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

def depth_scaled_init(model, n_layers=40):
    print("Applying Depth-Scaled Variance Initialization (GPT-style 1/sqrt(2L))...")
    std = 0.02
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=std)
            if "W_o.weight" in name or "W2.weight" in name:
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_layers))
        else:
            if p.requires_grad: nn.init.zeros_(p)

def run_8b_xavier_accumulator():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    print(f"\n>>> [TITAN 8B XAVIER-SCALED ACCUMULATOR] Preparing Model...")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0)
    
    # 1. Scaled Init
    depth_scaled_init(model, n_layers=40)
    
    # 2. Quantize to 4-bit (Safe Way)
    state_dict = model.state_dict()
    def replace_with_4bit(model):
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
    print(f">>> RUNNING 1000 STEPS (8B Xavier-Accumulator | Virtual BS={accum_steps})...")
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
                avg_t = (time.time()-start_time)/update_idx
                print(f"  Update {update_idx:4d}/1000 | Loss: {running_loss/accum_steps:.4f} | GN: {total_norm:.4f} | T/update: {avg_t:.2f}s")
                history.append({"step": update_idx, "loss": round(running_loss/accum_steps, 4), "gn": round(total_norm, 4)})
            
            running_loss = 0.0

if __name__ == "__main__":
    run_8b_xavier_accumulator()
