import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from model import PID8Transformer
import math
import torch.nn.functional as F

def get_activations(model, dataloader, layer_idx):
    model.eval()
    activations = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            for j in range(layer_idx + 1):
                x = model.layers[j](x)
            activations.append(x.view(-1, x.size(-1)))
            # Increase sample for large models (need at least d_model samples)
            if len(activations) * x.size(1) >= max(2000, model.d_model * 2): break 
    return torch.cat(activations, dim=0)

def get_dct_weights(out_dims, in_dims, warp=1.4):
    """Fully vectorized DCT-II basis calculation."""
    i = torch.arange(out_dims).view(-1, 1)
    j = torch.arange(in_dims).view(1, -1)
    warped_j = torch.pow(j / in_dims, warp) * in_dims
    W = torch.cos(math.pi / in_dims * (warped_j + 0.5) * i)
    return W * math.sqrt(2.0 / in_dims)

def init_phase0_embedding(model, dataloader, k=2000):
    """
    Phase 0: Lexical Seeding with Nyström Approximation.
    Allows full-vocabulary seeding with O(N*k) memory instead of O(N^2).
    """
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    k = min(k, vocab_size)
    
    # C_nk: Cross-correlations between ALL tokens and k LANDMARKS
    C_nk = torch.zeros(vocab_size, k, device='cpu')
    print(f"Seeding Embeddings (Phase 0 - Nyström Approximation, k={k})...")
    
    for i, batch in enumerate(dataloader):
        batch_cpu = batch.cpu()
        u = batch_cpu[:, :-1].reshape(-1)
        v = batch_cpu[:, 1:].reshape(-1)
        
        # We only care about co-occurrences where at least one token is a landmark
        mask_v_is_k = (v < k)
        mask_u_is_k = (u < k)
        
        # Case 1: v is a landmark, u is any token
        u1, v1 = u[mask_v_is_k], v[mask_v_is_k]
        if u1.numel() > 0:
            C_nk.index_put_((u1, v1), torch.ones_like(u1, dtype=torch.float), accumulate=True)
            
        # Case 2: u is a landmark, v is any token
        u2, v2 = u[mask_u_is_k], v[mask_u_is_k]
        if u2.numel() > 0:
            C_nk.index_put_((v2, u2), torch.ones_like(v2, dtype=torch.float), accumulate=True)
            
        if i >= 100: break

    # W is the k x k submatrix of correlations between landmarks only
    W = C_nk[:k, :].float()
    
    # SVD of the landmarks matrix W
    # W = U_w * S_w * V_w^T
    Uw, Sw, Vw = torch.svd(W)
    
    # Nyström Extension: Approx eigenvectors for the FULL matrix
    # U_full = C_nk * Uw * inv(Sw)
    eps = 1e-6
    inv_Sw = torch.diag(1.0 / torch.sqrt(Sw + eps))
    U_approx = torch.matmul(C_nk.float(), torch.matmul(Uw, inv_Sw))
    
    # Seed the entire embedding layer
    seed = torch.randn(vocab_size, d_model) * 0.02
    dims_to_seed = min(d_model, k)
    seed[:, :dims_to_seed] = U_approx[:, :dims_to_seed]
    
    model.embedding.weight.data = seed.to(model.embedding.weight.data.device)

def init_phase5_whitening(model, dataloader):
    X = get_activations(model, dataloader, len(model.layers)-1)
    mu = X.mean(dim=0)
    X_centered = X - mu
    C = torch.matmul(X_centered.t(), X_centered) / X.size(0)
    U, S, V = torch.svd(C)
    eps = 1e-5
    whitening_matrix = torch.matmul(U, torch.matmul(torch.diag(1.0 / torch.sqrt(S + eps)), U.t()))
    model.unembed.weight.data = torch.matmul(model.unembed.weight.data - mu.to(model.unembed.weight.device), whitening_matrix.to(model.unembed.weight.device))

def init_phase6_calibration(model, dataloader):
    model.eval()
    print("Calibrating LayerNorms (Phase 6 - Robust)...")
    with torch.no_grad():
        accum_scales = [torch.zeros(1, device=next(model.parameters()).device) for _ in model.layers]
        count = 0
        for batch in dataloader:
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            for i, layer in enumerate(model.layers):
                x = layer(x)
                accum_scales[i] += torch.sqrt(1.0 / (x.var() + 1e-6))
            count += 1
            if count >= 10: break
        for i, layer in enumerate(model.layers):
            layer.ln1.weight.data *= (accum_scales[i] / count)
            layer.ln2.weight.data *= (accum_scales[i] / count)

def initialize_pid8(model, dataloader, zipf_warp=1.0, spectral_gamma=0.25, morph_alpha=0.45,
                    use_phase0=True, use_cast=True, use_hunchback=True, use_whitening=False, use_calibration=True):
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    if use_phase0: init_phase0_embedding(model, dataloader)
    
    print("Phase 1: Collecting Activations for K-Means (8B Scale)...")
    X_init = get_activations(model, dataloader, -1)
    
    print("Phase 2: Computing K-Means Clusters...")
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_init.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    print("Phase 3: Computing Global Spectral Priors (SVD)...")
    X_centered = X_init - X_init.mean(dim=0)
    U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
    
    print(f"Phase 4: Sculpting {n_layers} Layers (DPI Manifold Construction)...")
    dct_cache = {}
    for l in range(n_layers):
        if l % 5 == 0 or l == n_layers - 1:
            print(f"  Sculpting Layer {l:2d}/{n_layers}...")
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        cast_f = (1.0 - 0.5 * math.sin(math.pi * progress)) if use_cast else 1.0
        current_gamma = spectral_gamma * cast_f * (1.05 if l % 2 != 0 else 0.95)
        svd_b = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        if use_hunchback:
            ws, wk = math.exp(-progress * 4.0), math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        else: ws, wk = 1.0 - progress, progress
        d_mlp = model.layers[l].mlp.W1.out_features
        if (d_mlp, model.d_model) not in dct_cache:
            dct_cache[(d_mlp, model.d_model)] = get_dct_weights(d_mlp, model.d_model, warp=zipf_warp).to(device)
        model.layers[l].mlp.W1.weight.data = (ws * dct_cache[(d_mlp, model.d_model)] + wk * svd_b.repeat(math.ceil(d_mlp/model.d_model), 1)[:d_mlp]) / (ws + wk)
        wqkv = (1-progress) * centers + progress * svd_b
        model.layers[l].attn.W_q.weight.data = wqkv
        model.layers[l].attn.W_k.weight.data = wqkv
        model.layers[l].attn.W_v.weight.data = wqkv
        M = torch.randn(model.d_model, model.d_model, device=device)
        Q, _ = torch.linalg.qr(M)
        model.layers[l].attn.W_o.weight.data = Q
        M2 = torch.randn(d_mlp, d_mlp, device=device)
        Q2, _ = torch.linalg.qr(M2)
        model.layers[l].mlp.W2.weight.data = Q2[:model.d_model, :]
    if use_whitening: 
        print("Phase 5: Whitening Final Layer...")
        init_phase5_whitening(model, dataloader)
    if use_calibration: init_phase6_calibration(model, dataloader)
    print(f"PID-14 Turbo Initialized.")
