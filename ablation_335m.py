import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer, count_parameters
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import os
import json
import time

class ArxivDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=20000):
        self.seq_len = seq_len
        cache_path = f"arxiv_bpe_20000.pt"
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

def run_ablation_335m(name, use_whitening):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    # 335M Architecture
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=1024, n_heads=16, d_mlp=4096, n_layers=24, dropout=0.1).to(device)
    model.gradient_checkpointing = True
    
    print(f"\n>>> [335M ABLATION] Run: {name} (Whitening={use_whitening})")
    
    dataset = ArxivDataset("arxiv.train.raw", tokenizer, seq_len=128, max_lines=20000)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    class SL:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for x, y in self.dl: yield x.to(device)
            
    initialize_pid8(model, SL(loader), use_whitening=use_whitening)
    
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    start_time = time.time()
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= 200: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 20 == 0 or step == 0:
            print(f"  Step {step+1:3d}/200 | Loss: {loss.item():.4f} | T: {time.time()-start_time:.1f}s")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    return history

if __name__ == "__main__":
    results = {}
    # Run WITH Whitening (The Scaling Guardian)
    results["full_dpi"] = run_ablation_335m("Full_DPI", use_whitening=True)
    torch.cuda.empty_cache(); time.sleep(10)
    
    # Run WITHOUT Whitening (The "Speed Demon" that might fail here)
    results["no_white"] = run_ablation_335m("No_White", use_whitening=False)
    
    os.makedirs("tests/Ablation_335M", exist_ok=True)
    with open("tests/Ablation_335M/ablation_335m_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n335M ABLATION COMPLETE.")
