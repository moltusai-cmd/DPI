import torch
import torch.nn as nn
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import time
import os
import json

class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=10000):
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

def run_1b_test(alpha_value):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    # 1.1B Model (24 Layers, 2048 Dim)
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=2048, n_heads=16, d_mlp=8192, n_layers=24, dropout=0.0).to(device)
    
    print(f"\n>>> [1.1B INVARIANCE TEST] Alpha = {alpha_value}...")
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=10000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) # Reduced BS
    
    class SimpleLoader:
        def __iter__(self):
            for i, (x, y) in enumerate(loader):
                if i >= 20: break
                yield x.to(device)
                
    initialize_pid8(model, SimpleLoader(), morph_alpha=alpha_value, use_whitening=False)
    
    # Cast to BF16 for training efficiency
    model = model.to(torch.bfloat16)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    for step, (x, y) in enumerate(loader):
        if step >= 100: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step+1) % 20 == 0 or step == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f}")
            history.append(loss.item())
            
    return history[-1]

if __name__ == "__main__":
    # Test Alpha Optimal
    loss_optimal = run_1b_test(0.45)
    
    # Test Alpha Non-Optimal
    loss_cold = run_1b_test(0.10)
    
    print("\n--- FINAL 1.1B INVARIANCE VERDICT ---")
    print(f"Alpha 0.45 (Optimal): {loss_optimal:.4f}")
    print(f"Alpha 0.10 (Cold):    {loss_cold:.4f}")
    print(f"Advantage: {loss_cold - loss_optimal:.4f}")
