import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
import os
import re
from collections import Counter
import math

class WikiDataset(Dataset):
    def __init__(self, file_path, vocab, seq_len=512, max_lines=10000):
        self.seq_len = seq_len
        self.vocab = vocab
        self.data = []
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                tokens = self.tokenize(line)
                self.data.extend([self.vocab.get(t, self.vocab['<unk>']) for t in tokens])
        
        self.num_samples = (len(self.data) - 1) // seq_len

    def tokenize(self, text):
        return re.findall(r"[\w']+|[.,!?;=]|@-@", text.lower())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def build_vocab(file_path, vocab_size=16384, max_lines=50000):
    print(f"Building vocab from {file_path}...")
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            tokens = re.findall(r"[\w']+|[.,!?;=]|@-@", line.lower())
            counter.update(tokens)
    
    most_common = counter.most_common(vocab_size - 2)
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

def xavier_init(model):
    print("Applying Xavier Uniform Initialization...")
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    
    model = PID8Transformer(vocab_size=16384).to(device)
    
    # Standard Xavier Init
    xavier_init(model)
    
    train_dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=20000)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    step = 0
    print("\nStarting Training (Xavier Init)...")
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        step += 1
        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")
        
        if step >= 100:
            break

if __name__ == "__main__":
    train()
