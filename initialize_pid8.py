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
        for batch in dataloader:
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            for i in range(layer_idx + 1):
                x = model.layers[i](x)
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= 1000: break
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
    count = 0
    for batch in dataloader:
        for seq in batch:
            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i+1]
                if u < vocab_size and v < vocab_size:
                    cooc[u, v] += 1
                    cooc[v, u] += 1
        count += 1
        if count > 50: break
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
    with torch.no_grad():
        for batch in dataloader:
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            for i, layer in enumerate(model.layers):
                x = layer(x)
                var_out = x.var()
                scale = torch.sqrt(1.0 / (var_out + 1e-6))
                layer.ln1.weight.data *= scale
                layer.ln2.weight.data *= scale
            break

def apply_spectral_blur(w):
    w_padded = F.pad(w.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode='replicate')
    kernel = torch.tensor([[[[0.25], [0.50], [0.25]]]], device=w.device)
    return F.conv2d(w_padded, kernel).squeeze(0).squeeze(0)

def get_random_rotation(dim, device, epsilon=0.05):
    """Generates a rotation matrix near identity to ensure topological divergence."""
    M = torch.randn(dim, dim, device=device) * epsilon
    I = torch.eye(dim, device=device)
    # Cayley transform to get an orthogonal matrix
    A = M - M.t()
    R = torch.matmul(I + A, torch.inverse(I - A))
    return R

def initialize_pid8(model, dataloader, zipf_warp=1.1, spectral_gamma=0.35, morph_alpha=0.35):
    """PID-13: The TDA & Entropy-Lens Edition.
    Features: Topological Rotation (Betti divergence) & Entropy Modulation (Expansion/Pruning).
    """
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    
    init_phase0_embedding(model, dataloader)
    
    X_init = get_activations(model, dataloader, -1)
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_init.cpu().numpy())
    base_centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    X_centered = X_init - X_init.mean(dim=0)
    U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))

    current_centers = base_centers
    for l in range(n_layers):
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        
        # --- FEATURE 1: TOPOLOGICAL ROTATION (TDA) ---
        # Rotate the semantic Voronoi space at each layer to prevent redundancy.
        rotation = get_random_rotation(model.d_model, device, epsilon=0.02)
        current_centers = torch.matmul(current_centers, rotation)
        
        # --- FEATURE 2: ENTROPY MODULATION (Lens) ---
        # High variance at start (Expansion), Low variance at end (Pruning/Élagage).
        # Factor goes from 1.5 (uncertainty) to 0.5 (certainty).
        entropy_factor = 1.5 - progress 
        
        # CAST Spectral Modulation
        cast_factor = 1.0 - 0.5 * math.sin(math.pi * progress)
        current_gamma = spectral_gamma * cast_factor * (1.05 if l % 2 != 0 else 0.95)
        
        layer_svd_basis = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        
        # ID Hunchback
        w_semantic = math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        w_syntax = math.exp(-progress * 4.0)
        
        # MLP Transition
        dct_w1 = get_dct_weights(model.layers[l].mlp.W1.out_features, model.d_model, warp=zipf_warp).to(device)
        svd_w1 = layer_svd_basis.repeat(math.ceil(model.layers[l].mlp.W1.out_features / model.d_model), 1)[:model.layers[l].mlp.W1.out_features]
        
        # Ricci Sparsity in the middle
        total_w = w_syntax + w_semantic
        w1_mixed = (w_syntax * dct_w1 + w_semantic * svd_w1) / total_w
        if 0.3 < progress < 0.7:
            n_blocks = 4
            mask = torch.zeros_like(w1_mixed)
            chunk_out = mask.size(0) // n_blocks
            chunk_in = mask.size(1) // n_blocks
            for b in range(n_blocks):
                mask[b*chunk_out:(b+1)*chunk_out, b*chunk_in:(b+1)*chunk_in] = 1.0
            w1_mixed = w1_mixed * torch.where(mask == 1.0, 1.0, 0.1)
        model.layers[l].mlp.W1.weight.data = w1_mixed
        
        # Attention with Entropy Scaling
        # Apply entropy_factor to Wq and Wk to flatten/sharpen attention distribution
        wq = ((1-progress) * current_centers + progress * layer_svd_basis) * entropy_factor
        wk = ((1-progress) * current_centers + progress * layer_svd_basis) * entropy_factor
        wv = (1-progress) * current_centers + progress * layer_svd_basis
        
        # Spectral Blur at the end
        if progress > 0.7:
            wq = apply_spectral_blur(wq)
            wk = apply_spectral_blur(wk)
            wv = apply_spectral_blur(wv)
            
        model.layers[l].attn.W_q.weight.data = wq
        model.layers[l].attn.W_k.weight.data = wk
        model.layers[l].attn.W_v.weight.data = wv
        
        # Output projections (QR + Heartbeat V2)
        residual_gain = 1.2 if l % 2 != 0 else 0.2
        M = torch.randn(model.d_model, model.d_model, device=device)
        Q, _ = torch.linalg.qr(M)
        model.layers[l].attn.W_o.weight.data = Q * residual_gain
        M2 = torch.randn(model.layers[l].mlp.W1.out_features, model.layers[l].mlp.W1.out_features, device=device)
        Q2, _ = torch.linalg.qr(M2)
        model.layers[l].mlp.W2.weight.data = Q2[:model.d_model, :] * residual_gain

    init_phase5_whitening(model, dataloader)
    init_phase6_calibration(model, dataloader)
    
    print(f"PID-13 (TDA & Entropy-Lens Edition) Initialization Complete.")
