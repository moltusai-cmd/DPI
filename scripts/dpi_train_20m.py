import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
from tokenizers import ByteLevelBPETokenizer

# Add src to path for imports
sys.path.append('src')
from model import PID8Transformer, count_parameters
from initialize_dpi import initialize_dpi

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        cache_path = f"wiki_bpe_{max_lines}.pt"
        if os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}")
            self.data = torch.load(cache_path)
        else:
            self.data = []
            print(f"Tokenizing {max_lines} lines...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    encoded = tokenizer.encode(line)
                    self.data.extend(encoded.ids)
            torch.save(self.data, cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, accumulation_steps=16):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        if i % 100 == 0:
            print(f"Step {i:5d}/{len(loader)} | Loss: {loss.item() * accumulation_steps:.4f}")
    return total_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Configuration (20M Architecture)
    vocab_size = 16384
    d_model = 320
    n_heads = 5
    d_mlp = 1280
    n_layers = 8
    
    # 2. Setup Tokenizer and Data
    tokenizer_path = "data/tokenizers/bpe_tokenizer"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    
    train_file = "data/raw/wiki.train.raw"
    dataset = WikiDataset(train_file, tokenizer, seq_len=128, max_lines=100000)
    
    batch_size = 64
    accumulation_steps = 64 # Total effective batch size = 64 * 64 = 4096
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model Initialization
    model = PID8Transformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        d_mlp=d_mlp, 
        n_layers=n_layers
    ).to(device)
    
    print(f"Model Parameters: {count_parameters(model)/1e6:.2f}M")
    print(f"Effective Batch Size: {batch_size * accumulation_steps}")
    
    # 4. DPI Initialization (v16.2)
    print("\nStarting DPI Initialization (v16.2)...")
    initialize_dpi(model, loader, mode="v16.2", spectral_gamma=0.25)
    torch.cuda.empty_cache()
    
    # 5. Save DPI'd Model
    dpi_save_path = "model_dpi_20m_v16_2.pt"
    torch.save(model.state_dict(), dpi_save_path)
    print(f"DPI model saved to {dpi_save_path}")
    
    # 6. Fine Training (1e-4, 2 epochs)
    print("\nStarting training (LR: 1e-4, 2 epochs, Cosine Scheduler)...")
    epochs = 2
    # total_steps should be based on number of optimizer steps
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Simple Cosine Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, scheduler=scheduler, accumulation_steps=accumulation_steps)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
    
    # 7. Save final model
    final_save_path = "model_finetuned_dpi_20m.pt"
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

if __name__ == "__main__":
    main()
