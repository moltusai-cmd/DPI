import torch
import torch.nn as nn
import sys
import os
import mup
import json
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer
from initialize_dpi import initialize_dpi

class FastArxivDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=5000):
        self.seq_len = seq_len
        self.data = []
        file_path = "data/raw/arxiv.train.raw"
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5000: break
                self.data.extend(tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_layer_gn(model):
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.detach().norm().item()
    return norms

def run_diagnostic():
    device = torch.device("cuda")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    dataset = FastArxivDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    cfg = dict(vocab_size=vocab_size, d_model=320, n_heads=5, n_layers=8, use_rope=True, use_mup=True)
    model = PID8Transformer(**cfg).to(device)
    base_model = PID8Transformer(**cfg)
    mup.set_base_shapes(model, base_model)
    
    # Initialize with DPI
    initialize_dpi(model, loader)
    
    optimizer = mup.MuAdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    print(f"🚀 Starting Deep-Sense Diagnostic (muP+DPI Hybrid)...")
    
    it = iter(loader)
    total_steps = 1000
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        # Collect layer-wise GN before optimizer step
        gn_layers = get_layer_gn(model)
        
        optimizer.step()
        
        if step % 20 == 0 or step == 1:
            # Focus on problematic layers: W_v and W_q in layer 4 (transition) and Layer 0
            # Also MLP W1
            v_gn = gn_layers.get('layers.4.attn.W_v.weight', 0)
            q_gn = gn_layers.get('layers.4.attn.W_q.weight', 0)
            mlp_gn = gn_layers.get('layers.4.mlp.W1.weight', 0)
            
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | GN_V4: {v_gn:.4f} | GN_Q4: {q_gn:.4f} | GN_MLP4: {mlp_gn:.4f}")
            history.append({
                "step": step, 
                "loss": loss.item(),
                "gn_layers": gn_layers
            })
            
    with open("results/diagnostic_hybrid_gn.json", "w") as f:
        json.dump(history, f)
    print("\n✅ Diagnostic Complete. Data saved to results/diagnostic_hybrid_gn.json")

if __name__ == "__main__":
    run_diagnostic()
