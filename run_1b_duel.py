import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer, count_parameters
from initialize_pid8 import initialize_pid8
import os
import json
import time
from tokenizers import ByteLevelBPETokenizer

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=50000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        cache_path = f"wiki_bpe_50000.pt"
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

def kaiming_init(model):
    for name, p in model.named_parameters():
        if p.dim() > 1: nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        else: nn.init.zeros_(p)

def tfixup_init(model, n_layers):
    L = n_layers
    for name, p in model.named_parameters():
        if p.dim() > 1:
            if 'embedding' in name: nn.init.normal_(p, std=1.0)
            elif 'W_o' in name or 'W2' in name: nn.init.zeros_(p)
            elif 'attn' in name or 'mlp' in name:
                std = 0.67 * (L ** (-2/3))
                nn.init.normal_(p, std=std)
            else: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def run_titan_test(name, init_type):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # 1B ARCHITECTURE: d_model=1536, L=32, heads=12
    model = PID8Transformer(
        vocab_size=vocab_size, 
        d_model=1536, 
        n_heads=12, 
        d_mlp=6144, 
        n_layers=32, 
        dropout=0.1
    ).to(device)
    
    # ENABLE VRAM SAVING
    model.gradient_checkpointing = True
    
    print(f"\n>>> [TITAN 1B] Run: {name} | Scale: {count_parameters(model)/1e6:.2f}M params")
    
    dataset = WikiDataset("wiki.train.raw", tokenizer, seq_len=128, max_lines=50000)
    loader = DataLoader(dataset, batch_size=1, shuffle=True) # Batch 1 for VRAM safety
    
    if init_type == "dpi":
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl: yield x.to(device)
        initialize_pid8(model, SL(loader), use_whitening=False)
    elif init_type == "xavier": xavier_init(model)
    elif init_type == "kaiming": kaiming_init(model)
    elif init_type == "tfixup": tfixup_init(model, n_layers=32)
    
    # Professional LLM Training Config: 8-bit AdamW to save 6GB of VRAM
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    steps = 200
    start_time = time.time()
    
    for step, (x, y) in enumerate(loader):
        if step >= steps: break
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        # USE BF16 FOR BLACKWELL SPEED & MEMORY
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step+1:3d}/200 | Loss: {loss.item():.4f} | T: {elapsed:.1f}s")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    return history

if __name__ == "__main__":
    import math
    results = {}
    
    # 1. DPI
    results["dpi"] = run_titan_test("DPI_PID14", "dpi")
    torch.cuda.empty_cache(); time.sleep(10)
    
    # 2. XAVIER
    results["xavier"] = run_titan_test("Xavier", "xavier")
    torch.cuda.empty_cache(); time.sleep(10)
    
    # 3. KAIMING
    results["kaiming"] = run_titan_test("Kaiming", "kaiming")
    torch.cuda.empty_cache(); time.sleep(10)
    
    # 4. T-FIXUP
    results["tfixup"] = run_titan_test("TFixup", "tfixup")
    
    os.makedirs("tests/Titan_1B_Duel", exist_ok=True)
    with open("tests/Titan_1B_Duel/titan_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nTITAN 1B DUEL COMPLETE.")
