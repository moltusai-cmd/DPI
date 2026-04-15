import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import mup
import numpy as np

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi

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

class TinyWiki(Dataset):
    def __init__(self, target_tokens=10000):
        self.seq_len = 128
        tokenizer = SimpleBPETokenizer(16384)
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        all_tokens = []
        for entry in dataset:
            all_tokens.extend(tokenizer.encode(entry["text"]))
            if len(all_tokens) >= target_tokens: break
        self.tokens = torch.tensor(all_tokens[:target_tokens], dtype=torch.long)
    def __len__(self): return len(self.tokens) // self.seq_len - 1
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len], self.tokens[start + 1 : start + self.seq_len + 1]

# --- EXPANDED SEMANTIC GROUPS (Stress Test) ---
SEMANTIC_GROUPS = {
    0: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "30", "40", "50", "100", "thousand", "million", "billion", "first", "second", "third", "fourth", "fifth", "last", "one", "two", "three", "four", "five"],
    1: ["San", "York", "Los", "London", "Paris", "Berlin", "Tokyo", "city", "State", "World", "National", "district", "America", "France", "Japan", "England", "region", "town", "village", "capital", "island", "river", "mountain", "south", "north", "east", "west"],
    2: ["the", "of", "and", "in", "to", "a", "is", "was", "for", "with", "on", "as", "by", "at", "from", "that", "it", "his", "her", "which", "their", "an", "were", "are", "but", "not", "this", "had", "have", "or", "been", "one", "all", "so"]
}

def get_probe_data(model, tokenizer, device):
    model.eval()
    X, Y = [], []
    with torch.no_grad():
        for label, words in SEMANTIC_GROUPS.items():
            for word in words:
                tokens = tokenizer.encode(word)
                if not tokens: continue
                token_tensor = torch.tensor([tokens], device=device)
                x = model.embedding(token_tensor)
                x = model.pos_encoding(x)
                for i in range(4): # Layer 3
                    x = model.layers[i](x)
                X.append(x[0, -1, :])
                Y.append(label)
    return torch.stack(X), torch.tensor(Y, device=device)

def train_stress_probe(X, Y, noise_std=0.3, bottleneck_dim=16, epochs=300):
    """
    Trains a linear probe under heavy noise and dimensionality bottleneck.
    """
    dim = X.size(1)
    num_classes = 3
    
    # 1. Inject Noise
    X_noisy = X + torch.randn_like(X) * noise_std
    
    # 2. Bottleneck Projection (Linear bottleneck)
    model = nn.Sequential(
        nn.Linear(dim, bottleneck_dim),
        nn.ReLU(),
        nn.Linear(bottleneck_dim, num_classes)
    ).to(X.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Split into train/test for the probe itself
    indices = torch.randperm(len(X))
    train_idx = indices[:int(0.7 * len(X))]
    test_idx = indices[int(0.7 * len(X)):]
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_noisy[train_idx])
        loss = criterion(logits, Y[train_idx])
        loss.backward()
        optimizer.step()
    
    # Final Accuracy on CLEAN Test Set (Measuring robustness)
    with torch.no_grad():
        preds = torch.argmax(model(X[test_idx]), dim=1)
        acc = (preds == Y[test_idx]).float().mean().item()
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🛰️ LEVEL 2.1: SEMANTIC PROBING STRESS TEST (Noise 30% + Bottleneck 16)")
    
    tokenizer = SimpleBPETokenizer(16384)
    loader = DataLoader(TinyWiki(target_tokens=20000), batch_size=32, shuffle=True)
    
    cfg = dict(vocab_size=16384, d_model=320, n_heads=10, d_mlp=1280, n_layers=6)
    base_model = PID8Transformer(vocab_size=16384, d_model=64, n_heads=4, d_mlp=256, n_layers=6).to(device)
    
    results = []
    # Run 5 trials to get a stable average
    for trial in range(3):
        trial_res = {}
        for mode in ["dpi", "xavier"]:
            torch.manual_seed(42 + trial)
            model = PID8Transformer(**cfg).to(device)
            mup.set_base_shapes(model, base_model)
            if mode == "dpi":
                initialize_dpi(model, loader, mode="v17.0")
            else:
                for p in model.parameters():
                    if p.dim() >= 2: nn.init.xavier_uniform_(p)
            
            X, Y = get_probe_data(model, tokenizer, device)
            acc = train_stress_probe(X, Y, noise_std=0.3, bottleneck_dim=16)
            trial_res[mode] = acc
        results.append(trial_res)
        print(f"  Trial {trial+1}: DPI={trial_res['dpi']*100:.1f}% | Xavier={trial_res['xavier']*100:.1f}%")

    avg_dpi = np.mean([r['dpi'] for r in results])
    avg_xav = np.mean([r['xavier'] for r in results])

    print("\n" + "="*60)
    print(f"🧬 STRESS TEST REPORT (Robustness to Noise & Compression)")
    print("-" * 60)
    print(f"DPI v17.0 Avg Accuracy : {avg_dpi*100:.2f}%")
    print(f"Xavier Avg Accuracy    : {avg_xav*100:.2f}%")
    print(f"Delta Robustness       : {(avg_dpi-avg_xav)*100:+.2f}%")
    print("="*60)
    
    if avg_dpi > avg_xav + 0.05:
        print(f"✅ SUCCESS: DPI structure is significantly more robust (+{(avg_dpi-avg_xav)*100:.1f}pts).")
    else:
        print("⚠️ INCONCLUSIVE: Both models degrade similarly under noise.")
    print("="*60)

if __name__ == "__main__":
    main()
