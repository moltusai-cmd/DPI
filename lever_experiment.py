import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import re
from collections import Counter
import math
import json
import itertools

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

def run_experiment(ma, sg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    train_dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=50000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    class SimpleLoader:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    # Fixed ZW at 1.1 (the best small polisher)
    initialize_pid8(model, SimpleLoader(train_loader), zipf_warp=1.1, spectral_gamma=sg, morph_alpha=ma)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    steps = 1000
    for step, (x, y) in enumerate(train_loader):
        if step >= steps: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return round(total_loss / steps, 4)

if __name__ == "__main__":
    ma_values = [0.25, 0.35, 0.50, 0.60]
    sg_values = [0.35, 0.45, 0.55]
    
    grid = list(itertools.product(ma_values, sg_values))
    print(f"Starting LEVER Experiment (MA/SG Plane, Fixed ZW=1.1)...")
    
    results = []
    for ma, sg in grid:
        loss = run_experiment(ma, sg)
        print(f"  MA: {ma:.2f} | SG: {sg:.2f} => Loss: {loss:.4f}")
        results.append({'ma': ma, 'sg': sg, 'loss': loss})
        
    results.sort(key=lambda x: x['loss'])
    print("\nLever Experiment Complete. Top results:")
    for r in results[:5]:
        print(r)
