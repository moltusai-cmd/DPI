import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import json
import time
import math
from tokenizers import ByteLevelBPETokenizer

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        cache_path = f"wiki_bpe_{max_lines}.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    self.data.extend(tokenizer.encode(line).ids)
            torch.save(self.data, cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_scheduler(optimizer, total_steps):
    warmup_steps = int(0.1 * total_steps)
    plateau_steps = int(0.4 * total_steps)
    cosine_steps = total_steps - warmup_steps - plateau_steps
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + plateau_steps: return 1.0
        else:
            progress = float(current_step - warmup_steps - plateau_steps) / float(max(1, cosine_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_ablation(name, flags):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    
    # 20M Model
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=100000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class SL:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    print(f"\n>>> Running Ablation: {name}")
    initialize_pid8(model, SL(loader), **flags)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(loader)
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 50 == 0:
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
        if (step + 1) % 400 == 0 or (step + 1) == 1:
            print(f"  Step {step+1:4d}/{total_steps} | Loss: {loss.item():.4f}")
            
    return history

if __name__ == "__main__":
    studies = [
        ("Full_PID14", {"use_phase0": True, "use_cast": True, "use_hunchback": True, "use_whitening": True, "use_calibration": True}),
        ("No_Phase0", {"use_phase0": False, "use_cast": True, "use_hunchback": True, "use_whitening": True, "use_calibration": True}),
        ("No_CAST", {"use_phase0": True, "use_cast": False, "use_hunchback": True, "use_whitening": True, "use_calibration": True}),
        ("No_Hunchback", {"use_phase0": True, "use_cast": True, "use_hunchback": False, "use_whitening": True, "use_calibration": True}),
        ("No_Whitening", {"use_phase0": True, "use_cast": True, "use_hunchback": True, "use_whitening": False, "use_calibration": True}),
        ("No_Calibration", {"use_phase0": True, "use_cast": True, "use_hunchback": True, "use_whitening": True, "use_calibration": False}),
    ]
    
    all_results = {}
    for name, flags in studies:
        all_results[name] = run_ablation(name, flags)
        
    with open("ablation_final_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nFinal Ablation Study Complete.")
