import torch
import torch.nn as nn
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer

def get_qk_alignment(model):
    alignments = []
    for l in range(len(model.layers)):
        layer = model.layers[l]
        W_q = layer.attn.W_q.weight.data
        W_k = layer.attn.W_k.weight.data
        # Cosine similarity between Q and K weights
        # (Using mean across all rows/cols for a global alignment metric)
        sim = torch.nn.functional.cosine_similarity(W_q, W_k, dim=1).mean().item()
        alignments.append(sim)
    return alignments

def get_spectral_decay(weight):
    # Compute SVD of a weight matrix
    U, S, V = torch.svd(weight.float())
    # Return S normalized by its max value
    return (S / S.max()).cpu().numpy()

def analyze_models(dpi_path, ft_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 20M Architecture
    params = dict(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8)
    
    model_dpi = PID8Transformer(**params).to(device)
    model_dpi.load_state_dict(torch.load(dpi_path, weights_only=True))
    
    model_ft = PID8Transformer(**params).to(device)
    model_ft.load_state_dict(torch.load(ft_path, weights_only=True))
    
    print("=== ANALYSIS: DPI vs. Fine-tuned (1 Epoch @ 1e-5) ===")
    
    # 1. QK-Alignment Shift
    align_dpi = get_qk_alignment(model_dpi)
    align_ft = get_qk_alignment(model_ft)
    
    print("\n[1] QK-Alignment per Layer:")
    print(f"{'Layer':<6} | {'DPI':<10} | {'FT':<10} | {'Shift':<10}")
    print("-" * 40)
    for i, (a_dpi, a_ft) in enumerate(zip(align_dpi, align_ft)):
        shift = a_ft - a_dpi
        print(f"{i:<6} | {a_dpi:10.4f} | {a_ft:10.4f} | {shift:10.4f}")

    # 2. Embedding/Unembed Manifold Shift
    emb_dpi = model_dpi.embedding.weight.data
    emb_ft = model_ft.embedding.weight.data
    emb_sim = torch.nn.functional.cosine_similarity(emb_dpi, emb_ft, dim=1).mean().item()
    print(f"\n[2] Lexical Manifold Retention (Embedding CosSim): {emb_sim:.4f}")
    
    unembed_dpi = model_dpi.unembed.weight.data
    unembed_ft = model_ft.unembed.weight.data
    un_sim = torch.nn.functional.cosine_similarity(unembed_dpi, unembed_ft, dim=1).mean().item()
    print(f"    Lexical Manifold Retention (Unembed CosSim): {un_sim:.4f}")

    # 3. Spectral Analysis (Layer 0 MLP and Layer 7 MLP)
    print("\n[3] Spectral Decay (MLP W1):")
    for l in [0, 7]:
        s_dpi = get_spectral_decay(model_dpi.layers[l].mlp.W1.weight.data)
        s_ft = get_spectral_decay(model_ft.layers[l].mlp.W1.weight.data)
        # Entropy of the singular value distribution as a proxy for "richness"
        ent_dpi = -(s_dpi * np.log(s_dpi + 1e-10)).sum()
        ent_ft = -(s_ft * np.log(s_ft + 1e-10)).sum()
        print(f"    Layer {l} | DPI Entropy: {ent_dpi:.2f} | FT Entropy: {ent_ft:.2f} | Diff: {ent_ft - ent_dpi:.2f}")

    # 4. Phase-Shift Integrity
    # DPI v16.2 expects a shift at layer 3-4 (42% of 8 layers)
    # Exploratory (0-3) should have higher alignment, Consolidated (4-7) near zero.
    # Actually initialize_dpi uses 0.40 * sin for exploratory and 0.0001 for consolidated.
    print("\n[4] Phase-Shift Integrity (Target: Peak at ~42% depth, then drop):")
    target_shift_layer = 8 // 2 # 4
    exploratory_avg_ft = np.mean(align_ft[:target_shift_layer])
    consolidated_avg_ft = np.mean(align_ft[target_shift_layer:])
    print(f"    Avg Alignment (Exploratory 0-3): {exploratory_avg_ft:.4f}")
    print(f"    Avg Alignment (Consolidated 4-7): {consolidated_avg_ft:.4f}")
    ratio = exploratory_avg_ft / (consolidated_avg_ft + 1e-6)
    print(f"    Exploratory/Consolidated Ratio: {ratio:.2f}x")

if __name__ == "__main__":
    analyze_models("model_dpi_20m_v16_2.pt", "model_finetuned_20m.pt")
