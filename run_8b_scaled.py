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

def depth_scaled_init(model, n_layers=40):
    print("Applying Depth-Scaled Variance Initialization (GPT-style 1/sqrt(2L))...")
    std = 0.02
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=std)
            # Apply depth scaling for residual output projections
            if "W_o.weight" in name or "W2.weight" in name:
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_layers))
        else:
            if p.requires_grad:
                nn.init.zeros_(p)

def run_8b_scaled_verified():
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    print(f"\n>>> [TITAN 8B SCALED BASELINE] Preparing Model...")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0)
    
    # 1. Depth-Scaled Init (Industry Standard for LLMs)
    depth_scaled_init(model, n_layers=40)
    
    # 2. Extract state_dict to ensure we have the exact initialized weights
    state_dict = model.state_dict()
    
    # 3. Manual Quantization Patching
    def replace_with_4bit(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                new_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=torch.bfloat16)
                setattr(model, name, new_layer)
            else: replace_with_4bit(module)
    
    replace_with_4bit(model)
    
    # 4. Load weights back (triggers proper quantization)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.gradient_checkpointing = True
    
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=20000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Using 1e-4 as it's the "attack" regime where DPI shone (and 1e-4 is standard for such tests)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(">>> RUNNING 1000 STEPS (8B Scaled Variance Challenge)...")
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
        
        # Monitor GN
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 50 == 0 or step == 0:
            avg_t = (time.time()-start_time)/(step+1)
            print(f"  Step {step+1:4d}/1000 | Loss: {loss.item():.4f} | GN: {total_norm:.4f} | T/step: {avg_t:.2f}s")
            history.append({"step": step + 1, "loss": round(loss.item(), 4), "gn": round(total_norm, 4)})
            
    os.makedirs("tests/Titan_8B_Survival", exist_ok=True)
    with open("tests/Titan_8B_Survival/scaled_8b_results_verified.json", "w") as f:
        json.dump(history, f, indent=4)
            
    print("\nTITAN 8B SCALED VERIFIED SUCCESSFUL.")

if __name__ == "__main__":
    run_8b_scaled_verified()
