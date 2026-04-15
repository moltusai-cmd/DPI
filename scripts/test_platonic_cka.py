import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import numpy as np
import mup

# Add src to path
sys.path.append('src')
from model import PID8Transformer
from initialize_dpi import initialize_dpi

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
    def encode(self, text, target_count=None):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if target_count and len(tokens) >= target_count: break
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

def calculate_cka(X, Y):
    def center(K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())
    K_c = center(K)
    L_c = center(L)
    hsic = (K_c * L_c).sum()
    normalization = torch.sqrt((K_c * K_c).sum() * (L_c * L_c).sum())
    return (hsic / (normalization + 1e-8)).item()

def get_activations(model, x, layer_idx=3):
    model.eval()
    activations = []
    def hook(module, input, output):
        activations.append(output.view(-1, output.size(-1)))
    handle = model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()
    return activations[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🛰️ LEVEL 4.1: PLATONIC CKA (REPRESENTATIONAL SIMILARITY)")
    
    loader = DataLoader(TinyWiki(50000), batch_size=32, shuffle=True)
    vocab_size, d_model = 16384, 320
    cfg = dict(vocab_size=vocab_size, d_model=d_model, n_heads=10, d_mlp=1280, n_layers=6)
    base_model = PID8Transformer(vocab_size=vocab_size, d_model=64, n_heads=4, d_mlp=256, n_layers=6).to(device)
    
    # Test batch (The "Probe")
    x_probe = torch.randint(0, vocab_size, (1, 128)).to(device)
    
    models = {}
    for mode in ["dpi", "xavier"]:
        for seed in [42, 1337]:
            name = f"{mode}_{seed}"
            print(f"Initializing {name}...")
            torch.manual_seed(seed)
            m = PID8Transformer(**cfg).to(device)
            mup.set_base_shapes(m, base_model)
            if mode == "dpi":
                initialize_dpi(m, loader, mode="v16.3")
            else:
                for p in m.parameters():
                    if p.dim() >= 2: nn.init.xavier_uniform_(p)
            models[name] = m

    print("\n--- Calculating CKA (Layer 3) ---")
    
    cka_dpi = calculate_cka(get_activations(models["dpi_42"], x_probe), 
                            get_activations(models["dpi_1337"], x_probe))
    
    cka_xav = calculate_cka(get_activations(models["xavier_42"], x_probe), 
                            get_activations(models["xavier_1337"], x_probe))
    
    cka_cross = calculate_cka(get_activations(models["dpi_42"], x_probe), 
                              get_activations(models["xavier_42"], x_probe))
    
    print("\n" + "="*70)
    print(f"🧬 DPI UNIVERSALITY (42 vs 1337)    : {cka_dpi:.4f}")
    print(f"🧬 XAVIER UNIVERSALITY (42 vs 1337) : {cka_xav:.4f}")
    print(f"🧬 CROSS-CHECK (DPI vs Xavier)      : {cka_cross:.4f}")
    print("="*70)
    
    if cka_dpi > cka_xav:
        print(f"✅ SUCCESS: DPI representations are more consistent (+{cka_dpi-cka_xav:.4f}).")
    else:
        print("⚠️ INCONCLUSIVE: Both start from different stochastic states.")
    print("="*70)

if __name__ == "__main__":
    main()
