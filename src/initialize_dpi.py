import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v16.2 - Genomic Ready (Gold Standard)
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

def init_phase0_lexical(model, dataloader):
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    device = next(model.parameters()).device
    print(f"  [Phase 0] Seeding Lexical Manifold (Exact SVD)...")
    C = torch.zeros(vocab_size * vocab_size, device=device)
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)): x = batch[0].to(device)
        else: x = batch.to(device)
        u, v = x[:, :-1].reshape(-1), x[:, 1:].reshape(-1)
        C.index_add_(0, u * vocab_size + v, torch.ones_like(u, dtype=torch.float, device=device))
        if i >= 300: break
    C = C.view(vocab_size, vocab_size)
    U, S, V = torch.svd_lowrank(C.float(), q=d_model, niter=10)
    model.embedding.weight.data[:, :min(d_model, vocab_size)] = U[:, :min(d_model, vocab_size)]
    model.embedding.weight.data = normalize_weight(model.embedding.weight.data, target_std=0.02)
    return U, V

def initialize_dpi(model, dataloader, spectral_gamma=0.25, use_calibration=True, mlp_jitter=0.02, mode="v16.2"):
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    phase_shift_layer = n_layers // 2
    
    U_lex, V_lex = init_phase0_lexical(model, dataloader)
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    print(f"  [Phase 2] Bootstrapping Mode: {mode.upper()} (Genomic Ready)...")
    
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
        W1.weight.data = normalize_weight(mlp_basis + torch.randn_like(mlp_basis) * (0.1 * progress))
        
        # 2. Attention: Uniform Alignment
        W_q = getattr(attn, 'W_q', None) or getattr(attn, 'q_proj', None)
        W_k = getattr(attn, 'W_k', None) or getattr(attn, 'k_proj', None)
        W_v = getattr(attn, 'W_v', None) or getattr(attn, 'v_proj', None)
        W_o = getattr(attn, 'W_o', None) or getattr(attn, 'o_proj', None)
        
        is_consolidated = (l >= phase_shift_layer)
        alignment = 0.40 * math.sin(math.pi * progress) if not is_consolidated else 0.0001
        
        if not is_consolidated:
            W_k.weight.data = normalize_weight(centers + 0.2 * svd_basis)
            W_v.weight.data = normalize_weight(svd_basis)
        else:
            W_k.weight.data = normalize_weight(svd_basis)
            W_v.weight.data = normalize_weight(svd_basis)
            
        W_q.weight.data = normalize_weight(alignment * W_k.weight.data + (1 - alignment) * svd_basis)
        
        # 3. Projections
        W_o.weight.data = normalize_weight(torch.randn_like(W_o.weight.data), target_std=(1.0/math.sqrt(2*n_layers)) * math.sqrt(1.0/model.d_model))
        W2.weight.data = normalize_weight(torch.randn_like(W2.weight.data), target_std=(1.0/math.sqrt(2*n_layers)) * math.sqrt(1.0/W2.in_features))

    if use_calibration:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)): x = batch[0].to(device)
                else: x = batch.to(device)
                x = model.embedding(x); x = model.pos_encoding(x)
                for j, layer in enumerate(model.layers):
                    x = layer(x); target_var = 1.0 + 0.5 * math.sin(math.pi * (j / n_layers))
                    scale = torch.sqrt(target_var / (x.var() + 1e-6))
                    layer.ln1.weight.data *= scale; layer.ln2.weight.data *= scale
                if i >= 5: break

    # 4. Zero-Wait Head
    unembed = getattr(model, 'unembed', None) or getattr(model, 'lm_head', None)
    if unembed:
        unembed.weight.data[:, :min(model.d_model, unembed.out_features)] = V_lex[:, :min(model.d_model, unembed.out_features)]
        unembed.weight.data = normalize_weight(unembed.weight.data, target_std=0.02)

    print(f"DPI-V16.2 Initialization (Genomic Ready) Complete.")
