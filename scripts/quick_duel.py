import torch
import torch.nn as nn
import sys
import os
import time
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

class FastWikiDataset(Dataset):
    def __init__(self, cache_path, seq_len=128):
        self.seq_len = seq_len
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file {cache_path} not found. Run a preprocessing script first.")
        self.data = torch.load(cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def run_mini_session(model, loader, mode_name, device, steps=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    print(f"\n🚀 Training {mode_name} for {steps} steps...")
    start_time = time.time()
    
    for i, (x, y) in enumerate(loader):
        if i >= steps: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"  [{mode_name}] Step {i+1:3d} | Loss: {loss.item():.4f}")
            
    end_time = time.time()
    return loss.item(), end_time - start_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_file = "checkpoints/wiki_bpe_100000.pt" # Use our existing cache
    vocab_file = "data/tokenizers/bpe_tokenizer/vocab.json"
    merges_file = "data/tokenizers/bpe_tokenizer/merges.txt"
    
    if not os.path.exists(cache_file):
        print(f"❌ Cache not found at {cache_file}. Please ensure files are in place.")
        return

    # Load Dataset
    dataset = FastWikiDataset(cache_file)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 1. DPI MODEL
    model_dpi = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    print("\n--- INITIALIZING DPI-14.1 ---")
    initialize_dpi(model_dpi, loader, warp_zeta=1.1, spectral_gamma=0.25)
    
    # 2. XAVIER MODEL
    model_xavier = PID8Transformer(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    print("\n--- INITIALIZING XAVIER BASELINE ---")
    xavier_init(model_xavier)
    
    # DUEL
    dpi_final_loss, dpi_time = run_mini_session(model_dpi, loader, "DPI-14.1", device)
    xavier_final_loss, xavier_time = run_mini_session(model_xavier, loader, "Xavier", device)
    
    print("\n" + "="*40)
    print("🏆 QUICK DUEL RESULTS (100 STEPS)")
    print("="*40)
    print(f"DPI-14.1 Final Loss: {dpi_final_loss:.4f}")
    print(f"Xavier   Final Loss: {xavier_final_loss:.4f}")
    print(f"DPI Advantage: {xavier_final_loss - dpi_final_loss:.4f} points")
    print(f"Time Taken: DPI={dpi_time:.1f}s, Xavier={xavier_time:.1f}s")
    print("="*40)

if __name__ == "__main__":
    main()
