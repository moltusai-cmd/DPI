import torch
import torch.nn as nn
import sys
import os
import math
import mup
import json
import time
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer
from initialize_dpi import initialize_dpi

class FastArxivDataset(Dataset):
    def __init__(self, tokenizer, seq_len=256, max_lines=100000):
        self.seq_len = seq_len
        self.data = []
        file_path = "data/raw/arxiv.train.raw"
        if not os.path.exists(file_path):
            print(f"ERROR: Dataset not found at {file_path}")
            # Fallback to creating dummy data if real one is missing for testing
            self.data = torch.randint(0, tokenizer.get_vocab_size(), (seq_len * 1000,)).tolist()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines: break
                    self.data.extend(tokenizer.encode(line).ids)
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"  Dataset loaded: {self.num_samples} samples.")
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.data[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

def get_effective_rank(W, threshold=0.01):
    if W.dim() > 2: W = W.view(-1, W.size(-1))
    s = torch.linalg.svdvals(W.detach().float())
    return (s > threshold * s[0]).sum().item()

def generate_sample(model, tokenizer, device, prompt="The fundamental theorem of", max_new_tokens=40):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n  [Generation] Prompt: {prompt}")
    generated = tokens
    for _ in range(max_new_tokens):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    
    text = tokenizer.decode(generated)
    print(f"  [Result]: {text}\n")
    return text

def run_session(name, loader, device, options, total_steps=1000):
    print(f"\n🚀 Starting {name} (350M Class)...")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    # 350M PARAMETERS ARCHITECTURE (Llama-style)
    cfg = dict(
        vocab_size=vocab_size, d_model=1024, n_heads=16, n_layers=24, d_mlp=4096,
        use_rope=True, use_mup_attn=True, use_mup_readout=True,
        use_rmsnorm=True, use_swiglu=True
    )
    model = PID8Transformer(**cfg).to(device)
    model.gradient_checkpointing = True 
    
    # Official muP Shape Registration
    base_cfg = cfg.copy(); base_cfg['d_model'] = 128; base_cfg['d_mlp'] = 512
    base_model = PID8Transformer(**base_cfg)
    mup.set_base_shapes(model, base_model)
    
    # Initialization
    start_init = time.time()
    if options.get('use_dpi', False):
        initialize_dpi(model, loader, mode="v16.3")
    else:
        print(f"  [muP] Applying Elite Scaling (Xavier)...")
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding): mup.init.normal_(m.weight, std=1.0)
            elif isinstance(m, mup.MuReadout): mup.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                mup.init.normal_(m.weight, std=1.0 / math.sqrt(fan_in))
    init_time = time.time() - start_init
    print(f"  Initialization completed in {init_time:.2f}s")
    
    # Optimizer
    optimizer = mup.MuAdamW(model.parameters(), lr=options['base_lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Scheduler
    warmup_steps = options.get('warmup_steps', 0)
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        return 1.0 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss()
    history = []
    it = iter(loader)
    model.train()
    
    for step in range(1, total_steps + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        
        # Calculate GN
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if step % 500 == 0 or step == 1:
            mid_layer = model.layers[12].attn.W_q.weight
            rank = get_effective_rank(mid_layer)
            print(f"  [{name}] Step {step:4d} | Loss: {loss.item():.4f} | GN: {gn:.2f} | Rank: {rank}")
            history.append({"step": step, "loss": round(loss.item(), 4), "rank": rank, "gn": round(gn, 2)})
            
    # SAVE MODEL
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{name.lower()}_350m.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved to {save_path}")
    
    # GENERATE SAMPLE
    generate_sample(model, tokenizer, device)
            
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "data/tokenizers/bpe_tokenizer_arxiv"
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt")
    dataset = FastArxivDataset(tokenizer, seq_len=256)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    total_steps = 10000 # Extended run for deep semantic validation
    base_lr = 1e-4
    dpi_lr = 8e-4
    
    # TEST A: DPI Pure (0 Warmup) - SUPERCHARGED LR
    res_dpi = run_session("DPI_V16_3_STRESS", loader, device, 
                         {'use_dpi': True, 'base_lr': dpi_lr, 'warmup_steps': 0}, 
                         total_steps=total_steps)
    
    # TEST B: Baseline Xavier (2k Warmup) - STANDARD LR
    res_xavier = run_session("XAVIER_BASELINE", loader, device, 
                            {'use_dpi': False, 'base_lr': base_lr, 'warmup_steps': 2000}, 
                            total_steps=total_steps)
    
    # Save Results
    results = {"dpi": res_dpi, "xavier": res_xavier}
    with open("results_sprint_350m_arxiv.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"🏆 SPRINT 5K RESULTS: DPI VS XAVIER (350M)")
    print("="*80)
    print(f"{'Step':<10} | {'Xavier (2k W)':<15} | {'DPI (0 W)':<15} | {'Rank DPI':<10}")
    print("-" * 80)
    for i in range(len(res_dpi)):
        d = res_dpi[i]; x = res_xavier[i]
        print(f"{d['step']:<10} | {x['loss']:<15.4f} | {d['loss']:<15.4f} | {d['rank']:<10}")
    print("="*80)

if __name__ == "__main__":
    main()
