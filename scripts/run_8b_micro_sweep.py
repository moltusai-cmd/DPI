import torch
import torch.nn as nn
import bitsandbytes as bnb
from model import PID8Transformer, count_parameters
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import os
import json
import time

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

def run_point(alpha_value):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    print(f"\n>>> [8B MICRO-SWEEP] Testing Alpha = {alpha_value}...")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0)
    
    # 1. Initialize with specific alpha
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=50000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    class SimpleLoader:
        def __iter__(self):
            for i, (x, y) in enumerate(loader):
                if i >= 50: break # Increased to 50 for K-Means at 8B
                yield x # Keep on CPU
                
    # Initialize on CPU first, then move to GPU/4-bit
    # (Actually initialize_pid8 works layer-by-layer, we'll do it on CPU for RAM)
    initialize_pid8(model, SimpleLoader(), morph_alpha=alpha_value, use_whitening=False)
    
    # 2. Convert to 4-bit and move to GPU
    def replace_with_4bit(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                new_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=torch.bfloat16)
                setattr(model, name, new_layer)
            else: replace_with_4bit(module)
    
    state_dict = model.state_dict()
    replace_with_4bit(model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.gradient_checkpointing = True
    
    # 3. Short Training (10 Updates)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    accum_steps = 32
    
    model.train()
    total_loss = 0
    start_time = time.time()
    optimizer.zero_grad()
    
    updates_done = 0
    for step, (x, y) in enumerate(loader):
        if updates_done >= 10: break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)) / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            updates_done += 1
            print(f"  Update {updates_done}/10 | Loss: {loss.item()*accum_steps:.4f}")
            total_loss = loss.item() * accum_steps
            
    return total_loss

if __name__ == "__main__":
    # Point A: The Sweet Spot
    loss_a = run_point(0.45)
    
    # Point B: The Cold Spot
    loss_b = run_point(0.10)
    
    print("\n--- FINAL MICRO-SWEEP VERDICT (8B) ---")
    print(f"Alpha 0.45 (Optimal) Loss: {loss_a:.4f}")
    print(f"Alpha 0.10 (Cold) Loss: {loss_b:.4f}")
    print(f"Transferability Delta: {loss_b - loss_a:.4f}")
