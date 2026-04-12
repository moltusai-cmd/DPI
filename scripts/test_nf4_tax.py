import torch
import torch.nn as nn
import bitsandbytes as bnb
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
from tokenizers import ByteLevelBPETokenizer
import time
import os
import json

class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len=128, max_lines=5000):
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

def run_experiment(mode):
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer_arxiv/vocab.json", "bpe_tokenizer_arxiv/merges.txt")
    model = PID8Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=2048, n_heads=16, d_mlp=8192, n_layers=24, dropout=0.0)
    
    # Initialize DPI on CPU
    dataset = ArxivDataset(tokenizer, seq_len=128, max_lines=5000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    class SimpleLoader:
        def __iter__(self):
            for i, (x, y) in enumerate(loader):
                if i >= 10: break
                yield x
                
    initialize_pid8(model, SimpleLoader(), use_whitening=False)
    
    if mode == "NF4":
        print("\n>>> Quantizing to 4-bit (NF4)...")
        state_dict = model.state_dict()
        def replace_with_4bit(model):
            for name, module in model.named_children():
                if isinstance(module, nn.Linear):
                    new_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=torch.bfloat16)
                    setattr(model, name, new_layer)
                else: replace_with_4bit(module)
        replace_with_4bit(model)
        model.load_state_dict(state_dict)
    else:
        print("\n>>> Using BF16 (Native)...")
        model = model.to(torch.bfloat16)

    model.to(device)
    model.gradient_checkpointing = True
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    for step, (x, y) in enumerate(loader):
        if step >= 50: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        # GN Measure
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        if (step+1) % 10 == 0:
            print(f"  Step {step+1:2d} | Loss: {loss.item():.4f} | GN: {total_norm:.2f}")
            history.append({"step": step+1, "loss": loss.item(), "gn": total_norm})
            
    return history[-1]

if __name__ == "__main__":
    res_bf16 = run_experiment("BF16")
    
    # Explicit Cleanup
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    time.sleep(2) # Give OS time to breathe
    
    res_nf4 = run_experiment("NF4")
    
    print("\n--- QUANTIZATION TAX VERDICT (Step 50) ---")
    print(f"Native BF16: Loss {res_bf16['loss']:.4f} | GN {res_bf16['gn']:.2f}")
    print(f"Quantized NF4: Loss {res_nf4['loss']:.4f} | GN {res_nf4['gn']:.2f}")
    print(f"Loss Delta: {res_nf4['loss'] - res_bf16['loss']:.4f}")
    print(f"Signal Degradation: {((res_bf16['gn'] - res_nf4['gn']) / res_bf16['gn']) * 100:.1f}%")
