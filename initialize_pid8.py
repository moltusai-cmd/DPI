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
            if i >= 10: break # Use 10 batches for better statistics
    return torch.cat(activations, dim=0)

def get_dct_weights(out_dims, in_dims, warp=1.4):
    W = torch.zeros(out_dims, in_dims)
    for i in range(out_dims):
        for j in range(in_dims):
            warped_j = math.pow(j / in_dims, warp) * in_dims
            W[i, j] = math.cos(math.pi / in_dims * (warped_j + 0.5) * i)
    return W * math.sqrt(2.0 / in_dims)

def init_phase0_embedding(model, dataloader):
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    cooc = torch.zeros(vocab_size, vocab_size)
    print("Seeding Embeddings (Phase 0)...")
    for i, batch in enumerate(dataloader):
        for seq in batch:
            for j in range(len(seq) - 1):
                u, v = seq[j], seq[j+1]
                if u < vocab_size and v < vocab_size:
                    cooc[u, v] += 1
                    cooc[v, u] += 1
        if i >= 50: break
    U, S, V = torch.svd(cooc[:2000, :2000].float())
    seed = torch.randn(vocab_size, d_model) * 0.02
    seed[:2000, :min(d_model, 2000)] = U[:, :min(d_model, 2000)]
    model.embedding.weight.data = seed.to(model.embedding.weight.device)

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
    """Robust Calibration over 10 batches."""
    model.eval()
    print("Calibrating LayerNorms (Phase 6)...")
    with torch.no_grad():
        accum_scales = [torch.zeros(1, device=next(model.parameters()).device) for _ in model.layers]
        count = 0
        for batch in dataloader:
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            for i, layer in enumerate(model.layers):
                x = layer(x)
                var_out = x.var()
                accum_scales[i] += torch.sqrt(1.0 / (var_out + 1e-6))
            count += 1
            if count >= 10: break
            
        for i, layer in enumerate(model.layers):
            final_scale = accum_scales[i] / count
            layer.ln1.weight.data *= final_scale
            layer.ln2.weight.data *= final_scale

def initialize_pid8(model, dataloader, zipf_warp=1.1, spectral_gamma=0.35, morph_alpha=0.35):
    """PID-14: Rigorous Manifold Edition.
    Clean implementation of CAST + Hunchback + Phase 0 + Robust Calibration.
    """
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    
    init_phase0_embedding(model, dataloader)
    
    X_init = get_activations(model, dataloader, -1)
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_init.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    X_centered = X_init - X_init.mean(dim=0)
    U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))

    for l in range(n_layers):
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        
        # 1. CAST Spectral Modulation
        cast_factor = 1.0 - 0.5 * math.sin(math.pi * progress)
        current_gamma = spectral_gamma * cast_factor * (1.05 if l % 2 != 0 else 0.95)
        layer_svd_basis = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        
        # 2. ID Hunchback
        w_semantic = math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        w_syntax = math.exp(-progress * 4.0)
        
        # MLP: Rigorous Blend
        dct_w1 = get_dct_weights(model.layers[l].mlp.W1.out_features, model.d_model, warp=zipf_warp).to(device)
        svd_w1 = layer_svd_basis.repeat(math.ceil(model.layers[l].mlp.W1.out_features / model.d_model), 1)[:model.layers[l].mlp.W1.out_features]
        model.layers[l].mlp.W1.weight.data = (w_syntax * dct_w1 + w_semantic * svd_w1) / (w_syntax + w_semantic)
        
        # Attention: Symmetrical Manifold
        wqkv = (1-progress) * centers + progress * layer_svd_basis
        model.layers[l].attn.W_q.weight.data = wqkv
        model.layers[l].attn.W_k.weight.data = wqkv
        model.layers[l].attn.W_v.weight.data = wqkv
        
        # Output Projections: Pure Orthogonality (Isometry)
        M = torch.randn(model.d_model, model.d_model, device=device)
        Q, _ = torch.linalg.qr(M)
        model.layers[l].attn.W_o.weight.data = Q
        
        M2 = torch.randn(model.layers[l].mlp.W1.out_features, model.layers[l].mlp.W1.out_features, device=device)
        Q2, _ = torch.linalg.qr(M2)
        model.layers[l].mlp.W2.weight.data = Q2[:model.d_model, :]

    init_phase5_whitening(model, dataloader)
    init_phase6_calibration(model, dataloader)
    
    print(f"PID-14 (Rigorous Manifold) Initialization Complete.")
