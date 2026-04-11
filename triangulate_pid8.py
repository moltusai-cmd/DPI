import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import json
import time
import itertools
from tokenizers import ByteLevelBPETokenizer

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        cache_path = f"wiki_bpe_{max_lines}.pt"
        self.data = torch.load(cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 50: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return round(total_loss / 50, 4)

def run_triangulation_point(params):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    
    # Standard 20M Model
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=320, n_heads=5, d_mlp=1280, n_layers=8, dropout=0.1).to(device)
    
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=100000)
    indices = list(range(len(dataset)))
    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=32, shuffle=False)
    
    class SL:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    initialize_pid8(model, SL(train_loader), 
                    zipf_warp=params['zipf_warp'], 
                    spectral_gamma=params['spectral_gamma'], 
                    morph_alpha=params['morph_alpha'],
                    use_whitening=False) # We know whitening is better OFF at this scale
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # 1 Epoch training
    total_steps = len(train_loader)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    final_val_loss = evaluate(model, val_loader, device)
    return final_val_loss

if __name__ == "__main__":
    test_dir = "tests/Triangulation_Final_1Epoch"
    os.makedirs(test_dir, exist_ok=True)
    
    zipf_warps = [1.0, 1.2, 1.4]
    spectral_gammas = [0.25, 0.35, 0.45]
    morph_alphas = [0.25, 0.35, 0.45]
    
    grid = list(itertools.product(zipf_warps, spectral_gammas, morph_alphas))
    
    print(f"Starting Final Triangulation (1 Epoch per point, {len(grid)} points)...")
    results = []
    
    for zw, sg, ma in grid:
        p = {'zipf_warp': zw, 'spectral_gamma': sg, 'morph_alpha': ma}
        start_t = time.time()
        val_loss = run_triangulation_point(p)
        elapsed = time.time() - start_t
        print(f"  ZW: {zw:.1f} | SG: {sg:.2f} | MA: {ma:.2f} => Val Loss: {val_loss:.4f} ({elapsed:.1f}s)")
        results.append({'params': p, 'val_loss': val_loss})
        
    results.sort(key=lambda x: x['val_loss'])
    with open(f"{test_dir}/triangulation_1epoch.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nTriangulation Complete. Best Point:")
    print(results[0])
