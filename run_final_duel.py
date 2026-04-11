import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import re
from collections import Counter
import math
import json
import time

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

def xavier_init(model):
    for name, p in model.named_parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def run_training(init_type, log_path, checkpoint_path):
    device = torch.device("cuda")
    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    
    # 60M Model: d_model=512, d_mlp=2048, n_layers=14, n_heads=8
    model = PID8Transformer(vocab_size=16384, d_model=512, n_heads=8, d_mlp=2048, n_layers=14).to(device)
    
    from model import count_parameters
    print(f"\nModel Scale: {count_parameters(model) / 1e6:.2f}M parameters")
    
    dataset = WikiDataset(train_file, vocab, seq_len=128, max_lines=100000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    if init_type == "dpi":
        print("\n>>> STARTING DPI TRAINING (Sweet Spot: ZW=1.1, SG=0.55, MA=0.50)")
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl: yield x.to(device)
        initialize_pid8(model, SL(loader), zipf_warp=1.1, spectral_gamma=0.55, morph_alpha=0.50)
    else:
        print("\n>>> STARTING XAVIER TRAINING (Standard Baseline)")
        xavier_init(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 10
    total_steps = len(loader) * epochs
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    global_step = 0
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            epoch_loss += loss.item()
            
            if global_step % 10 == 0:
                history.append({
                    "step": global_step,
                    "loss": round(loss.item(), 4),
                    "lr": float(f"{optimizer.param_groups[0]['lr']:.2e}")
                })
            
            if global_step % 200 == 0:
                elapsed = time.time() - start_time
                print(f"[{init_type.upper()}] Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
                
    # Save Results
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)
    torch.save(model.state_dict(), checkpoint_path)
    print(f">>> {init_type.upper()} Complete. Log: {log_path} | Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    # DPI FIRST
    run_training("dpi", "logs_dpi.json", "model_dpi_final.pt")
    
    # CLEAR MEMORY
    torch.cuda.empty_cache()
    time.sleep(5)
    
    # XAVIER SECOND
    run_training("xavier", "logs_xavier.json", "model_xavier_final.pt")
    
    print("\nDUEL COMPLETE. Good night!")
