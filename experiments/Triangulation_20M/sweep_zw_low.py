import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import re
from collections import Counter
import math
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

def run_experiment(zw):
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
            
    initialize_pid8(model, SimpleLoader(train_loader), zipf_warp=zw, spectral_gamma=0.35, morph_alpha=0.35)
    
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
    zw_values = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    print(f"Starting ZW Extended Sweep (Fixed SG=0.35, MA=0.35)...")
    results = []
    for zw in zw_values:
        loss = run_experiment(zw)
        print(f"  ZW: {zw:.1f} => Loss: {loss:.4f}")
        results.append({'zw': zw, 'loss': loss})
        
    print("\nSweep Complete.")
    results.sort(key=lambda x: x['loss'])
    print(f"Best ZW: {results[0]['zw']} with Loss {results[0]['loss']}")
