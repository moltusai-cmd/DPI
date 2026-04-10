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

def get_scheduler(optimizer, total_steps):
    warmup_steps = int(0.1 * total_steps)
    plateau_steps = int(0.4 * total_steps)
    cosine_steps = total_steps - warmup_steps - plateau_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + plateau_steps:
            return 1.0
        else:
            progress = float(current_step - warmup_steps - plateau_steps) / float(max(1, cosine_steps))
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    
    model = PID8Transformer(vocab_size=16384).to(device)
    
    # Xavier Initialization
    xavier_init(model)
    
    # Same dataset and loader as PID-8.1 test
    train_dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=100000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    epochs = 4
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"\nStarting {epochs} Epochs Training (Xavier Init, {total_steps} total steps)...")
    print(f"Schedule: 10% Warm-up, 40% Plateau (1e-4), 50% Cosine Decay (-> 1e-5)")
    
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0 or global_step == 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step:4d}/{total_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        avg_loss = epoch_loss / steps_per_epoch
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")
        
    print("Training Complete.")
    torch.save(model.state_dict(), "xavier_4epochs.pt")

if __name__ == "__main__":
    train()
