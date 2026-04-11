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

def init_phase0_embedding(model, dataloader):
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    cooc = torch.zeros(vocab_size, vocab_size, device='cpu')
    print("Seeding Embeddings (Phase 0 - Vectorized)...")
    for i, batch in enumerate(dataloader):
        # batch is on GPU, bring to CPU for co-occurrence matrix
        batch_cpu = batch.cpu()
        u = batch_cpu[:, :-1].reshape(-1)
        v = batch_cpu[:, 1:].reshape(-1)
        mask = (u < vocab_size) & (v < vocab_size)
        u, v = u[mask], v[mask]
        cooc.index_put_((u, v), torch.ones_like(u, dtype=torch.float), accumulate=True)
        cooc.index_put_((v, u), torch.ones_like(v, dtype=torch.float), accumulate=True)
        if i >= 100: break
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

def initialize_pid8(model, dataloader, zipf_warp=1.1, spectral_gamma=0.35, morph_alpha=0.35,
                    use_phase0=True, use_cast=True, use_hunchback=True, use_whitening=True, use_calibration=True):
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    if use_phase0: init_phase0_embedding(model, dataloader)
    X_init = get_activations(model, dataloader, -1)
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_init.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    X_centered = X_init - X_init.mean(dim=0)
    U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
    dct_cache = {}
    for l in range(n_layers):
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
    if use_whitening: init_phase5_whitening(model, dataloader)
    if use_calibration: init_phase6_calibration(model, dataloader)
    print(f"PID-14 Turbo Initialized.")
