import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import math
import time

class WikiDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=20000):
        self.seq_len = seq_len
        self.data = []
        with open("wiki.train.raw", 'r', encoding='utf-8') as f:
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

def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_duel(mode):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    dataset = WikiDataset(tokenizer, seq_len=128, max_lines=20000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    if mode == "DPI-Warmup":
        print("\n>>> Running DPI (10% Warmup)...")
        class SimpleLoader:
            def __iter__(self):
                for x, y in loader: yield x.to(device)
        initialize_pid8(model, SimpleLoader(), use_whitening=False)
    else:
        print("\n>>> Running Xavier (10% Industrial Warmup)...")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)

    scheduler = get_warmup_scheduler(optimizer, int(0.1 * len(loader)))
    history = []
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= 300: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (step+1) % 100 == 0:
            print(f"  Step {step+1:4d} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            history.append(loss.item())
            
    return history

if __name__ == "__main__":
    res_dpi = run_duel("DPI-Warmup")
    res_xavier = run_duel("Xavier-Warmup")
    
    print("\n--- FINAL VERDICT (Step 300) ---")
    print(f"DPI (10% Warmup) Loss: {res_dpi[-1]:.4f}")
    print(f"Xavier (10% Warmup) Loss: {res_xavier[-1]:.4f}")
    print(f"Advantage: {res_xavier[-1] - res_dpi[-1]:.4f}")
