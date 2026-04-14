import torch
import torch.nn as nn
import mup
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer

def verify():
    d_base = 160
    d_target = 320 # 2x multiplier
    vocab_size = 16384
    
    # Target Model
    cfg = dict(vocab_size=vocab_size, d_model=d_target, n_heads=5, n_layers=2, use_mup_attn=True, use_mup_readout=True)
    model = PID8Transformer(**cfg)
    
    # Base Model for muP
    base_cfg = cfg.copy(); base_cfg['d_model'] = d_base; base_cfg['d_mlp'] = d_base * 4
    base_model = PID8Transformer(**base_cfg)
    
    mup.set_base_shapes(model, base_model)
    
    print("🔬 INSPECTING mup.init.normal_(std=0.02) BEHAVIOR:")
    print("-" * 50)
    
    # Initialize
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding, mup.MuReadout)):
            mup.init.normal_(m.weight, std=0.02)
            std = m.weight.std().item()
            print(f"Layer: {name:<25} | Actual STD: {std:.6f}")

if __name__ == "__main__":
    verify()
