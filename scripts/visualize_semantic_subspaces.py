import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import mup

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi
from optimizer import DPISpectralOptimizer

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.inv_vocab = {}
    def encode(self, text):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if h not in self.inv_vocab: self.inv_vocab[h] = word
        return tokens
    def decode(self, token_id):
        return self.inv_vocab.get(token_id, f"[{token_id}]")

class TinyWiki(Dataset):
    def __init__(self, tokenizer, target_tokens=1_000_000):
        self.seq_len = 128
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        all_tokens = []
        print("📦 Populating Vocabulary...")
        for entry in dataset:
            all_tokens.extend(tokenizer.encode(entry["text"]))
            if len(all_tokens) >= target_tokens: break
        self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🛰️ LEVEL 1: DIRECT SEMANTIC VISUALIZATION")
    
    tokenizer = SimpleBPETokenizer(16384)
    # Increase sample size for better vocabulary coverage
    loader = DataLoader(TinyWiki(tokenizer, 1_000_000), batch_size=32, shuffle=True)
    
    cfg = dict(vocab_size=16384, d_model=320, n_heads=10, d_mlp=1280, n_layers=6)
    base_model = PID8Transformer(vocab_size=16384, d_model=64, n_heads=4, d_mlp=256, n_layers=6).to(device)
    model = PID8Transformer(**cfg).to(device)
    mup.set_base_shapes(model, base_model)
    
    # Initialize with DPI v17.0
    initialize_dpi(model, loader, mode="v17.0")
    
    optimizer = DPISpectralOptimizer(model.parameters(), lr=8e-4, anchor_factor=0.42)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Training 200 steps to crystallize semantic axes ---")
    model.train()
    steps = 0
    while steps < 200:
        for x, y in loader:
            if steps >= 200: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1, 16384), y.view(-1))
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 50 == 0: print(f"  Step {steps:4d} | Loss: {loss.item():.4f}")

    # --- VISUALIZATION CORE ---
    print("\n--- Extracting Dominant Semantic Directions (Layer 3) ---")
    with torch.no_grad():
        W = model.layers[3].attn.W_q.weight.data
        U, S, V = torch.svd(W)
        
        k = 10 # Let's see more directions
        top_directions = V[:, :k] 
        E = model.embedding.weight.data 
        scores = torch.matmul(E, top_directions) 
        
        for i in range(k):
            print(f"\n💎 Direction {i} (Sigma={S[i]:.2f}):")
            _, top_indices = scores[:, i].topk(15)
            words = [tokenizer.decode(idx.item()) for idx in top_indices]
            print(f"  AXIS+: {', '.join(words)}")
            
            _, bot_indices = scores[:, i].topk(15, largest=False)
            words_bot = [tokenizer.decode(idx.item()) for idx in bot_indices]
            print(f"  AXIS-: {', '.join(words_bot)}")

if __name__ == "__main__":
    main()
