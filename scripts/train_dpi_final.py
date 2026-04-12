import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import re
from collections import Counter
import math
import csv

class WikiDataset(Dataset):
    def __init__(self, file_path, vocab, seq_len=128, max_lines=100000):
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

def train():
    device = torch.device("cuda")
    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    model = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=100000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class SL:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    print("Initializing DPI (Sweet Spot)...")
    initialize_pid8(model, SL(loader), zipf_warp=1.1, spectral_gamma=0.55, morph_alpha=0.50)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 10
    total_steps = len(loader) * epochs
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    log_file = open("loss_dpi.csv", "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "loss", "lr"])
    
    global_step = 0
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            if global_step % 10 == 0:
                logger.writerow([global_step, round(loss.item(), 4), optimizer.param_groups[0]['lr']])
                log_file.flush()
            if global_step % 100 == 0:
                print(f"DPI | Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f}")
                
    torch.save(model.state_dict(), "model_dpi_final.pt")
    log_file.close()

if __name__ == "__main__":
    train()
