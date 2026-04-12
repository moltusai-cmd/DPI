import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import json
import time
from tokenizers import ByteLevelBPETokenizer

class ArxivDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        cache_path = f"arxiv_bpe_{max_lines}.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    self.data.extend(tokenizer.encode(line).ids)
            torch.save(self.data, cache_path)
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def xavier_init(model):
    for name, p in model.named_parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def run_300m_experiment(name, init_mode, use_whitening=True):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    model = PID8Transformer(
        vocab_size=16384, 
        d_model=1024, 
        n_heads=16, 
        d_mlp=4096, 
        n_layers=24, 
        dropout=0.1
    ).to(device)
    
    from model import count_parameters
    print(f"\n>>> Model: {name} | Scale: {count_parameters(model)/1e6:.2f}M params")
    
    dataset = ArxivDataset("arxiv.train.raw", tokenizer, seq_len=128, max_lines=100000)
    # 16 workers for parallel data loading
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
    
    if init_mode == "dpi":
        print(f"Initializing DPI (use_whitening={use_whitening})...")
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl: yield x.to(device)
        initialize_pid8(model, SL(loader), use_whitening=use_whitening)
    else:
        print("Initializing Xavier Baseline...")
        xavier_init(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    steps = 1000
    start_time = time.time()
    
    for step, (x, y) in enumerate(loader):
        if step >= steps: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - start_time
            print(f"  [{name}] Step {step+1:4d}/1000 | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
            
    return history

if __name__ == "__main__":
    # Test 1: Full PID-14
    full_res = run_300m_experiment("Full_DPI", "dpi", use_whitening=True)
    torch.cuda.empty_cache(); time.sleep(5)
    
    # Test 2: DPI No Whitening
    no_white_res = run_300m_experiment("No_Whitening", "dpi", use_whitening=False)
    torch.cuda.empty_cache(); time.sleep(5)
    
    # Test 3: Xavier Baseline
    xavier_res = run_300m_experiment("Xavier_Baseline", "xavier")
    
    with open("ablation_300m_results.json", "w") as f:
        json.dump({
            "full_dpi": full_res, 
            "no_whitening": no_white_res,
            "xavier": xavier_res
        }, f, indent=4)
    
    print("\n300M Heavyweight Comparison Complete.")
