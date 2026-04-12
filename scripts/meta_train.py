import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import re
from collections import Counter
import math
import time
import json

class WikiDataset(Dataset):
    def __init__(self, file_path, vocab, seq_len=128, max_lines=50000):
        self.seq_len = seq_len
        self.vocab = vocab
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                tokens = re.findall(r"[\w']+|[.,!?;=]|@-@", line.lower())
                self.data.extend([self.vocab.get(t, self.vocab['<unk>']) for t in tokens])
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def build_vocab(file_path, vocab_size=16384, max_lines=50000):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            tokens = re.findall(r"[\w']+|[.,!?;=]|@-@", line.lower())
            counter.update(tokens)
    most_common = counter.most_common(vocab_size - 2)
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common)}
    vocab['<pad>'] = 0; vocab['<unk>'] = 1
    return vocab

def run_experiment(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    
    # 20M Model
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    train_dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=100000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    class SimpleLoader:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    print(f"\nExperiment: {params}")
    initialize_pid8(model, SimpleLoader(train_loader), 
                    zipf_warp=params['zipf_warp'], 
                    spectral_gamma=params['spectral_gamma'], 
                    morph_alpha=params['morph_alpha'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    model.train()
    for step, (x, y) in enumerate(train_loader):
        if step >= 1000: break
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if step in [0, 99, 499, 999]:
            results[f"step_{step+1}"] = round(loss.item(), 4)
            print(f"  Step {step+1:4d} | Loss: {loss.item():.4f}")
            
    return results

if __name__ == "__main__":
    experiments = [
        {'zipf_warp': 1.0, 'spectral_gamma': 0.5, 'morph_alpha': 0.2}, # Baseline
        {'zipf_warp': 1.5, 'spectral_gamma': 0.5, 'morph_alpha': 0.2}, # Zipf+
        {'zipf_warp': 0.5, 'spectral_gamma': 0.5, 'morph_alpha': 0.2}, # Zipf-
        {'zipf_warp': 1.0, 'spectral_gamma': 0.8, 'morph_alpha': 0.2}, # Gamma+
        {'zipf_warp': 1.0, 'spectral_gamma': 0.2, 'morph_alpha': 0.2}, # Gamma-
        {'zipf_warp': 1.0, 'spectral_gamma': 0.5, 'morph_alpha': 0.5}, # Morph+
        {'zipf_warp': 1.2, 'spectral_gamma': 0.4, 'morph_alpha': 0.3}, # Mixed 1
        {'zipf_warp': 0.8, 'spectral_gamma': 0.6, 'morph_alpha': 0.1}, # Mixed 2
        {'zipf_warp': 1.5, 'spectral_gamma': 0.3, 'morph_alpha': 0.4}, # Mixed 3
        {'zipf_warp': 2.0, 'spectral_gamma': 0.5, 'morph_alpha': 0.2}, # Extreme Zipf
    ]
    
    all_results = []
    for p in experiments:
        res = run_experiment(p)
        all_results.append({'params': p, 'results': res})
        
    with open("meta_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nMeta-Experiment Complete. Results saved to meta_results.json")
