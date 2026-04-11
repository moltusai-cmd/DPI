import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import json
from tokenizers import ByteLevelBPETokenizer

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=50000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                self.data.extend(self.tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def run_ablation(name, flags):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=50000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class SL:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    print(f"\n>>> Running Ablation: {name}")
    initialize_pid8(model, SL(loader), **flags)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    model.train()
    steps = 1000
    for step, (x, y) in enumerate(loader):
        if step >= steps: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        history.append({"step": step + 1, "loss": round(loss.item(), 4)})
        if (step + 1) % 200 == 0:
            print(f"  Step {step+1:4d} | Loss: {loss.item():.4f}")
            
    return history

if __name__ == "__main__":
    studies = [
        ("Full_DPI", {"use_phase0": True, "use_cast": True, "use_heartbeat": True, "use_hunchback": True, "use_qr": True}),
        ("No_CAST", {"use_phase0": True, "use_cast": False, "use_heartbeat": True, "use_hunchback": True, "use_qr": True}),
        ("No_Heartbeat", {"use_phase0": True, "use_cast": True, "use_heartbeat": False, "use_hunchback": True, "use_qr": True}),
        ("No_Hunchback", {"use_phase0": True, "use_cast": True, "use_heartbeat": True, "use_hunchback": False, "use_qr": True}),
        ("No_Phase0", {"use_phase0": False, "use_cast": True, "use_heartbeat": True, "use_hunchback": True, "use_qr": True}),
        ("No_QR", {"use_phase0": True, "use_cast": True, "use_heartbeat": True, "use_hunchback": True, "use_qr": False}),
        ("Minimalist_DCT_SVD", {"use_phase0": False, "use_cast": False, "use_heartbeat": False, "use_hunchback": False, "use_qr": False}),
    ]
    
    all_ablation_results = {}
    for name, flags in studies:
        history = run_ablation(name, flags)
        all_ablation_results[name] = history
        
    with open("ablation_results.json", "w") as f:
        json.dump(all_ablation_results, f, indent=4)
        
    print("\n=== Ablation Study Complete. Data saved to ablation_results.json ===")
