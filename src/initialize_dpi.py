import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v16.0 - Multi-Generation Support (v15.2 & v16.0)
"""

def get_activations(model, dataloader, layer_idx, num_samples=2000):
    model.eval()
    activations = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)): x_batch = batch[0].to(device)
            else: x_batch = batch.to(device)
            x = model.embedding(x_batch); x = model.pos_encoding(x)
            for j in range(layer_idx + 1): x = model.layers[j](x)
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= num_samples: break 
    return torch.cat(activations, dim=0)

def normalize_weight(W, target_std=None):
    if target_std is None: target_std = math.sqrt(1.0 / W.size(1))
    curr_std = W.std()
    if curr_std > 1e-8: return W * (target_std / curr_std)
    return W

def initialize_dpi(model, dataloader, spectral_gamma=0.25, use_calibration=True, mlp_jitter=0.02, mode="v16"):
    """
    Deterministic Pipeline Initialization (DPI)
    Core Engine v16.0 - Phase-Shift Genomic
    Args:
        mode: "v16" (Phase-Shift Genomic - Recommended) or "v15" (Hyper-Resonance)
    """
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    phase_shift_layer = n_layers // 2
    
    print(f"  [Phase 0] Seeding Lexical Manifold...")
    # (Phase 0 logic omitted for brevity, same as v15.2)
    
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    print(f"  [Phase 2] Bootstrapping Mode: {mode.upper()}...")
    
    for l in range(n_layers):
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        current_gamma = spectral_gamma * (1.0 - 0.2 * math.sin(math.pi * progress))
        svd_basis = normalize_weight((U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device))
        
        layer = model.layers[l]
        attn = getattr(layer, 'attn', None) or getattr(layer, 'attention', None)
        mlp = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
        
        # 1. MLP Init
        W1 = getattr(mlp, 'W1', None) or getattr(mlp, 'fc1', None)
        W2 = getattr(mlp, 'W2', None) or getattr(mlp, 'fc2', None)
        mlp_basis = svd_basis.repeat(W1.out_features // model.d_model, 1)
        
        if mode == "v16":
            compression_noise = torch.randn_like(mlp_basis) * (0.1 * progress)
            W1.weight.data = normalize_weight(mlp_basis + compression_noise)
        else:
            W1.weight.data = normalize_weight(mlp_basis)
            
        if mlp_jitter > 0: W1.weight.data += torch.randn_like(W1.weight.data) * mlp_jitter
        
        # 2. Attention Init
        W_q = getattr(attn, 'W_q', None) or getattr(attn, 'q_proj', None)
        W_k = getattr(attn, 'W_k', None) or getattr(attn, 'k_proj', None)
        W_v = getattr(attn, 'W_v', None) or getattr(attn, 'v_proj', None)
        W_o = getattr(attn, 'W_o', None) or getattr(attn, 'o_proj', None)
        
        if mode == "v16":
            is_consolidated = (l >= phase_shift_layer)
            if not is_consolidated:
                alignment = 0.4 * math.sin(math.pi * (l / phase_shift_layer))
                W_k.weight.data = normalize_weight(centers + 0.2 * svd_basis)
                W_v.weight.data = normalize_weight(svd_basis)
            else:
                alignment = 0.0001
                shared_manifold = normalize_weight(svd_basis)
                W_k.weight.data = shared_manifold
                W_v.weight.data = shared_manifold
        else: # v15.2 Hyper-Resonance
            alignment = 0.4 * math.sin(math.pi * progress)
            W_k.weight.data = normalize_weight(centers + 0.2 * svd_basis)
            W_v.weight.data = normalize_weight(svd_basis)
            
        W_q.weight.data = normalize_weight(alignment * W_k.weight.data + (1-alignment) * svd_basis)
        
        # 3. Output Projections
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        W_o.weight.data = normalize_weight(torch.randn_like(W_o.weight.data), target_std=res_scale * math.sqrt(1.0/model.d_model))
        W2.weight.data = normalize_weight(torch.randn_like(W2.weight.data), target_std=res_scale * math.sqrt(1.0/W2.in_features))

    if use_calibration:
        print("  [Phase 3] Final Manifold Calibration...")
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)): x = batch[0].to(device)
                else: x = batch.to(device)
                x = model.embedding(x); x = model.pos_encoding(x)
                for j, layer in enumerate(model.layers):
                    x = layer(x)
                    target_var = 1.0 + (0.5 * math.sin(math.pi * (j / n_layers)) if mode=="v16" else 0.0)
                    scale = torch.sqrt(target_var / (x.var() + 1e-6))
                    layer.ln1.weight.data *= scale; layer.ln2.weight.data *= scale
                if i >= 5: break
    print(f"DPI-{mode.upper()} Initialization Complete.")
