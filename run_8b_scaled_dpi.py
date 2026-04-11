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
    def __init__(self, tokenizer, seq_len=128, max_lines=20000):
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

def apply_depth_scaling_to_dpi(state_dict, n_layers=40):
    scale_factor = 1.0 / math.sqrt(2 * n_layers)
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

def run_8b_high_speed_hybrid():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    print(f"\n>>> [TITAN 8B S-DPI HIGH-SPEED] Target LR: 5e-5")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0)
    
    state_dict = torch.load("dpi_8b_bf16.pt", map_location='cpu', weights_only=True)
    state_dict = apply_depth_scaling_to_dpi(state_dict, n_layers=40)
    
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
    
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=20000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)
    scheduler = get_scheduler(optimizer, warmup_steps=100)
    criterion = nn.CrossEntropyLoss()
    
    print(">>> RUNNING 1000 STEPS (S-DPI High-Speed Challenge)...")
    history = []
    start_time = time.time()
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= 1000: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:4d}/1000 | Loss: {loss.item():.4f} | GN: {total_norm:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})

if __name__ == "__main__":
    run_8b_high_speed_hybrid()
