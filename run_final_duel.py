import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import PID8Transformer
from initialize_pid8 import initialize_pid8
import os
import json
import time
from tokenizers import ByteLevelBPETokenizer

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128, max_lines=100000):
        self.seq_len = seq_len
        dataset_name = os.path.basename(file_path).split('.')[0]
        cache_path = f"{dataset_name}_bpe_{max_lines}.pt"
        
        if os.path.exists(cache_path):
            print(f"Loading tokenized data from cache: {cache_path}")
            self.data = torch.load(cache_path)
        else:
            self.data = []
            print(f"Loading and tokenizing {file_path} with BPE...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    tokens = tokenizer.encode(line).ids
                    self.data.extend(tokens)
            print(f"Saving tokenized data to cache: {cache_path}")
            torch.save(self.data, cache_path)
            
        self.num_samples = (len(self.data) - 1) // seq_len
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_scheduler(optimizer, total_steps):
    warmup_steps = int(0.1 * total_steps)
    plateau_steps = int(0.4 * total_steps)
    cosine_steps = total_steps - warmup_steps - plateau_steps
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + plateau_steps: return 1.0
        else:
            progress = float(current_step - warmup_steps - plateau_steps) / float(max(1, cosine_steps))
            import math
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def xavier_init(model):
    for name, p in model.named_parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.zeros_(p)

def run_training(init_type, log_path, checkpoint_path):
    device = torch.device("cuda")
    train_file = "wiki.train.raw"
    
    # Load BPE Tokenizer for WikiText
    tokenizer = ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # 20M Model
    model = PID8Transformer(vocab_size=vocab_size, d_model=320, n_heads=5, d_mlp=1280, n_layers=8).to(device)
    
    from model import count_parameters
    print(f"\nModel Scale: {count_parameters(model) / 1e6:.2f}M parameters | Vocab: {vocab_size}")
    
    dataset = WikiDataset(train_file, tokenizer, seq_len=128, max_lines=100000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    if init_type == "dpi":
        print("\n>>> STARTING DPI TRAINING (PID-13 TDA & Entropy-Lens)")
        class SL:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl: yield x.to(device)
        initialize_pid8(model, SL(loader), zipf_warp=1.1, spectral_gamma=0.55, morph_alpha=0.50)
    else:
        print("\n>>> STARTING XAVIER TRAINING (Standard Baseline)")
        xavier_init(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 1
    total_steps = len(loader) * epochs
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    global_step = 0
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            if global_step % 10 == 0:
                history.append({
                    "step": global_step,
                    "loss": round(loss.item(), 4),
                    "lr": float(f"{optimizer.param_groups[0]['lr']:.2e}")
                })
            
            if global_step % 200 == 0:
                elapsed = time.time() - start_time
                print(f"[{init_type.upper()}] Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
                
    # Save Results
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)
    torch.save(model.state_dict(), checkpoint_path)
    print(f">>> {init_type.upper()} Complete. Log: {log_path} | Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    # DPI FIRST
    run_training("dpi", "logs_dpi_wiki_20m.json", "model_dpi_wiki_20m.pt")
    
    # CLEAR MEMORY
    torch.cuda.empty_cache()
    time.sleep(5)
    
    # XAVIER SECOND
    run_training("xavier", "logs_xavier_wiki_20m.json", "model_xavier_wiki_20m.pt")
    
    print("\nWIKI 20M DUEL COMPLETE.")
