import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer, count_parameters
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import os
import json
import time
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, cpu_offload
import bitsandbytes as bnb

class ArxivDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=2000):
        self.seq_len = seq_len
        self.data = []
        with open("arxiv.train.raw", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                self.data.extend(tokenizer.encode(line).ids)
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

def run_8b_survival(name, init_type):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    
    # TITAN 8B ARCHITECTURE (~8.2B params)
    # d_model: 4096, layers: 40, d_mlp: 16384
    print(f"\n>>> [TITAN 8B SURVIVAL] Init: {name}")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=4096, n_heads=32, d_mlp=16384, n_layers=40, dropout=0.0).to("cpu")
    print(f"  Model Scale: {count_parameters(model)/1e9:.2f}B parameters")
    
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=2000)
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # Small batch for 8B
    
    if init_type == "dpi":
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                # KEEP DATA ON CPU during DPI Init because the 8B model is on CPU
                for x, y in self.dl: yield x.to("cpu")
        # DPI on CPU (will take ~2-3 mins but it's 8B)
        initialize_pid8(model, SL(loader), use_whitening=False)
        print(f"  DPI Initialized. Saving 8B weights to disk...")
        torch.save(model.state_dict(), "dpi_8b_init.pt")
        print(f"  DPI Weights Saved: dpi_8b_init.pt")
    else:
        xavier_init(model)
        print(f"  Xavier Initialized. Saving 8B weights to disk...")
        torch.save(model.state_dict(), "xavier_8b_init.pt")
        print(f"  Xavier Weights Saved: xavier_8b_init.pt")
        
    # Moving model to CPU Offload (using Accelerate)
    # This will load layers into GPU only when needed
    model = cpu_offload(model, execution_device=device)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    start_time = time.time()
    for step, (x, y) in enumerate(loader):
        if step >= 10: break # Only 10 steps for survival test
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"  Step {step+1:2d}/10 | Loss: {loss.item():.4f} | T: {time.time()-start_time:.1f}s")
        history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    return history

if __name__ == "__main__":
    results = {}
    # DPI Run
    results["dpi"] = run_8b_survival("DPI_PID14", "dpi")
    torch.cuda.empty_cache(); time.sleep(10)
    
    # Xavier Run
    results["xavier"] = run_8b_survival("Xavier", "xavier")
    
    os.makedirs("tests/Titan_8B_Survival", exist_ok=True)
    with open("tests/Titan_8B_Survival/survival_8b_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nTITAN 8B SURVIVAL TEST COMPLETE.")
