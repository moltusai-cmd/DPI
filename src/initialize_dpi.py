import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v14.1 - Sequential Bootstrapping
"""

def get_activations(model, dataloader, layer_idx, num_samples=2000):
    model.eval()
    activations = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)): x_batch = batch[0].to(device)
            else: x_batch = batch.to(device)
            x = model.embedding(x_batch)
            x = model.pos_encoding(x)
            for j in range(layer_idx + 1):
                x = model.layers[j](x)
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= num_samples:
                break 
    return torch.cat(activations, dim=0)

def get_dct_weights(out_dims, in_dims, warp=1.4):
    i = torch.arange(out_dims).view(-1, 1)
    j = torch.arange(in_dims).view(1, -1)
    warped_j = torch.pow(j / in_dims, warp) * in_dims
    W = torch.cos(math.pi / in_dims * (warped_j + 0.5) * i)
    return W * math.sqrt(2.0 / in_dims)

def normalize_weight(W, target_std=None):
    if target_std is None:
        target_std = math.sqrt(1.0 / W.size(1))
    curr_std = W.std()
    if curr_std > 1e-8:
        return W * (target_std / curr_std)
    return W

def init_phase0_embedding(model, dataloader, k_samples=2000, use_exact_svd=False):
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    device = next(model.parameters()).device

    if use_exact_svd:
        print(f"  [Phase 0] Seeding Lexical Manifold (Iterative SVD via {device})...")
        C = torch.zeros(vocab_size * vocab_size, device=device)
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)): x = batch[0].to(device)
            else: x = batch.to(device)
            u, v = x[:, :-1].reshape(-1), x[:, 1:].reshape(-1)
            idx = u * vocab_size + v
            C.index_add_(0, idx, torch.ones_like(u, dtype=torch.float, device=device))
            if i >= 300: break
        C = C.view(vocab_size, vocab_size)
        U, S, V = torch.svd_lowrank(C.float(), q=d_model, niter=10)
        model.embedding.weight.data[:, :min(d_model, vocab_size)] = U[:, :min(d_model, vocab_size)]
    else:
        k = min(k_samples, vocab_size)
        C_nk = torch.zeros(vocab_size, k, device='cpu')
        print(f"  [Phase 0] Seeding Lexical Manifold (Nyström SVD)...")
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)): x = batch[0]
            else: x = batch
            u, v = x[:, :-1].reshape(-1), x[:, 1:].reshape(-1)
            mask = (v < k)
            u_m, v_m = u[mask], v[mask]
            if u_m.numel() > 0:
                C_nk.index_put_((u_m, v_m), torch.ones_like(u_m, dtype=torch.float), accumulate=True)
            if i >= 100: break
        W = C_nk[:k, :].float()
        Uw, Sw, Vw = torch.svd(W)
        inv_Sw = torch.diag(1.0 / torch.sqrt(Sw + 1e-6))
        U_approx = torch.matmul(C_nk.float(), torch.matmul(Uw, inv_Sw))
        model.embedding.weight.data[:, :min(d_model, k)] = U_approx[:, :min(d_model, k)].to(device)
    
    model.embedding.weight.data = normalize_weight(model.embedding.weight.data, target_std=0.02)

def initialize_dpi(model, dataloader, warp_zeta=1.0, spectral_gamma=0.25, morph_alpha=0.45, use_calibration=True, use_exact_svd=False, residual_scale=1.0):
    """
    Main entry point for DPI-14.1 Sequential Bootstrapping.
    
    Args:
        residual_scale: Scaling factor for output projections (Standard: 1.0, Deep Models: 1/sqrt(2L)).
    """
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    
    init_phase0_embedding(model, dataloader, use_exact_svd=use_exact_svd)
    
    print("  [Phase 1] Computing Global Lexical Centers...")
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    print(f"  [Phase 2] Sequential Bootstrapping of {n_layers} Layers (ResScale: {residual_scale:.3f})...")
    dct_cache = {}
    
    for l in range(n_layers):
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        current_gamma = spectral_gamma * (1.0 - 0.5 * math.sin(math.pi * progress))
        
        svd_basis = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        svd_basis = normalize_weight(svd_basis)
        
        # --- MLP Pre-conditioning ---
        ws, wk = math.exp(-progress * 4.0), math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        d_mlp = model.layers[l].mlp.W1.out_features
        if (d_mlp, model.d_model) not in dct_cache:
            dct_cache[(d_mlp, model.d_model)] = get_dct_weights(d_mlp, model.d_model, warp=warp_zeta).to(device)
        
        mlp_init = (ws * dct_cache[(d_mlp, model.d_model)] + wk * svd_basis.repeat(math.ceil(d_mlp/model.d_model), 1)[:d_mlp]) / (ws + wk)
        model.layers[l].mlp.W1.weight.data = normalize_weight(mlp_init)
        
        # --- Functional QKV Signatures ---
        ortho_peak = math.sin(math.pi * progress)
        M_k = (1-progress) * centers + progress * svd_basis
        Q_k, _ = torch.linalg.qr(M_k.t())
        
        model.layers[l].attn.W_k.weight.data = normalize_weight((1-ortho_peak) * M_k + ortho_peak * Q_k.t())
        
        svd_v = (U.t() * torch.pow(S + 1e-6, current_gamma * 0.4).unsqueeze(1)).to(device)
        model.layers[l].attn.W_v.weight.data = normalize_weight(svd_v)
        
        alignment = 0.6 * (1.0 - progress)
        q_init = alignment * model.layers[l].attn.W_k.weight.data + (1-alignment) * svd_basis
        model.layers[l].attn.W_q.weight.data = normalize_weight(q_init)
        
        # Orthogonal projections with Optional Residual Scaling
        Q_o, _ = torch.linalg.qr(torch.randn(model.d_model, model.d_model, device=device))
        model.layers[l].attn.W_o.weight.data = normalize_weight(Q_o, target_std=residual_scale * math.sqrt(1.0/model.d_model))
        
        Q2, _ = torch.linalg.qr(torch.randn(d_mlp, d_mlp, device=device))
        model.layers[l].mlp.W2.weight.data = normalize_weight(Q2[:model.d_model, :], target_std=residual_scale * math.sqrt(1.0/d_mlp))
        
        if l % 5 == 0 or l == n_layers - 1:
            print(f"    Layer {l:2d} Sculpted (Signal Energy: {S[0].item():.2f})")

    if use_calibration: 
        print("  [Phase 3] Final Manifold Calibration (LayerNorm Scaling)...")
        model.eval()
        with torch.no_grad():
            accum_scales = [torch.zeros(1, device=device) for _ in model.layers]
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)): x_batch = batch[0].to(device)
                else: x_batch = batch.to(device)
                x = model.embedding(x_batch)
                x = model.pos_encoding(x)
                for j, layer in enumerate(model.layers):
                    x = layer(x)
                    accum_scales[j] += torch.sqrt(1.0 / (x.var() + 1e-6))
                if i >= 10: break
            for i, layer in enumerate(model.layers):
                scale = torch.clamp(accum_scales[i] / 11, 0.1, 2.0)
                layer.ln1.weight.data *= scale
                layer.ln2.weight.data *= scale
                
    print(f"DPI-14.1 Sequential Bootstrapping Complete.")
