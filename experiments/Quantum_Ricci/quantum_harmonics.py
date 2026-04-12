import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import math
import re
from collections import Counter

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

def run_quantum_experiment(name, ma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    
    # Ratio 4:1:1 based on MA
    sg = ma
    zw = 4 * sg
    
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    train_dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=50000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    class SimpleLoader:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    initialize_pid8(model, SimpleLoader(train_loader), zipf_warp=zw, spectral_gamma=sg, morph_alpha=ma)
    
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
        
    avg_loss = total_loss / steps
    print(f"  {name} | MA: {ma:.4f} | SG: {sg:.4f} | ZW: {zw:.4f} => Avg Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    H = 5
    experiments = [
        ("sqrt(2)/H", math.sqrt(2) / H),
        ("sqrt(3)/H", math.sqrt(3) / H),
        ("phi/H", (1 + math.sqrt(5)) / 2 / H),
        ("pi/9 (Ref)", math.pi / 9),
        ("2/H (Death Zone)", 2.0 / H)
    ]
    
    print(f"Starting Quantum Harmonic Validation (H={H})...")
    for name, ma in experiments:
        run_quantum_experiment(name, ma)
