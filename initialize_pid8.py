import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from model import PID8Transformer
import math
import torch.nn.functional as F

def get_activations(model, dataloader, layer_idx, num_samples=2000):
    """Collect activations after layer_idx (-1 means after embedding)."""
    model.eval()
    activations = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)): batch = batch[0] # Unpack x
            x = model.embedding(batch.to(device)) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            # Forward only up to the current layer
            for j in range(layer_idx + 1):
                x = model.layers[j](x)
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= num_samples: break 
    return torch.cat(activations, dim=0)

def get_dct_weights(out_dims, in_dims, warp=1.4):
    i = torch.arange(out_dims).view(-1, 1)
    j = torch.arange(in_dims).view(1, -1)
    warped_j = torch.pow(j / in_dims, warp) * in_dims
    W = torch.cos(math.pi / in_dims * (warped_j + 0.5) * i)
    return W * math.sqrt(2.0 / in_dims)

def init_phase0_embedding(model, dataloader, k=2000):
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    C_nk = torch.zeros(vocab_size, min(k, vocab_size), device='cpu')
    print(f"Seeding Embeddings (Phase 0 - Nyström)...")
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)): batch = batch[0] # Handle [x, y] list
        u = batch[:, :-1].reshape(-1); v = batch[:, 1:].reshape(-1)
        mask = (v < k)
        u1, v1 = u[mask], v[mask]
        if u1.numel() > 0: C_nk.index_put_((u1, v1), torch.ones_like(u1, dtype=torch.float), accumulate=True)
        if i >= 100: break
    W = C_nk[:k, :].float()
    Uw, Sw, Vw = torch.svd(W)
    inv_Sw = torch.diag(1.0 / torch.sqrt(Sw + 1e-6))
    U_approx = torch.matmul(C_nk.float(), torch.matmul(Uw, inv_Sw))
    model.embedding.weight.data[:, :min(d_model, k)] = U_approx[:, :min(d_model, k)].to(model.embedding.weight.device)

def initialize_pid8(model, dataloader, zipf_warp=1.0, spectral_gamma=0.25, morph_alpha=0.45, use_calibration=True):
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    
    # 1. Phase 0: Lexical Seed
    init_phase0_embedding(model, dataloader)
    
    # 2. Global Proto-Clusters (Dictionary)
    print("Phase 1: Computing Global Lexical Centers...")
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    # 3. Sequential Manifold Sculpting (The Core Fix)
    print(f"Phase 2: Sequential Sculpting of {n_layers} Layers (No Circularity)...")
    dct_cache = {}
    
    for l in range(n_layers):
        # COLLECT ACTIVATIONS FROM PREVIOUS LAYER (Real signal flow)
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        # SVD of CURRENT manifold
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        cast_f = (1.0 - 0.5 * math.sin(math.pi * progress))
        # Note: Previous versions used an alternating jitter (1.05/0.95) to break inter-layer symmetry.
        # PID-14.1 replaces this empirical 'hack' with functional QKV signatures (orthogonality, 
        # manifold deployment, and routing alignment) which provide natural geometric differentiation.
        current_gamma = spectral_gamma * cast_f
        svd_b = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        
        # MLP Sculpting
        ws, wk = math.exp(-progress * 4.0), math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        d_mlp = model.layers[l].mlp.W1.out_features
        if (d_mlp, model.d_model) not in dct_cache:
            dct_cache[(d_mlp, model.d_model)] = get_dct_weights(d_mlp, model.d_model, warp=zipf_warp).to(device)
        
        model.layers[l].mlp.W1.weight.data = (ws * dct_cache[(d_mlp, model.d_model)] + wk * svd_b.repeat(math.ceil(d_mlp/model.d_model), 1)[:d_mlp]) / (ws + wk)
        
        # QKV Functional Signatures
        ortho_peak = math.sin(math.pi * progress)
        M_k = (1-progress) * centers + progress * svd_b
        Q_k, _ = torch.linalg.qr(M_k.t()) 
        model.layers[l].attn.W_k.weight.data = (1-ortho_peak) * M_k + ortho_peak * Q_k.t()
        
        svd_v = (U.t() * torch.pow(S + 1e-6, current_gamma * 0.4).unsqueeze(1)).to(device)
        model.layers[l].attn.W_v.weight.data = svd_v
        
        alignment = 0.6 * (1.0 - progress)
        model.layers[l].attn.W_q.weight.data = alignment * model.layers[l].attn.W_k.weight.data + (1-alignment) * svd_b
        
        # Projections
        Q_o, _ = torch.linalg.qr(torch.randn(model.d_model, model.d_model, device=device))
        model.layers[l].attn.W_o.weight.data = Q_o
        Q2, _ = torch.linalg.qr(torch.randn(d_mlp, d_mlp, device=device))
        model.layers[l].mlp.W2.weight.data = Q2[:model.d_model, :]
        
        if l % 5 == 0 or l == n_layers - 1:
            print(f"  Layer {l:2d} Sculpted (SVD Energy: {S[0].item():.2f})")

    if use_calibration: 
        print("Phase 3: Final Manifold Calibration...")
        model.eval()
        with torch.no_grad():
            accum_scales = [torch.zeros(1, device=device) for _ in model.layers]
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)): batch = batch[0] # Unpack x
                x = model.embedding(batch.to(device)) * math.sqrt(model.d_model)
                x = model.pos_encoding(x)
                for j, layer in enumerate(model.layers):
                    x = layer(x)
                    accum_scales[j] += torch.sqrt(1.0 / (x.var() + 1e-6))
                if i >= 10: break
            for i, layer in enumerate(model.layers):
                layer.ln1.weight.data *= (accum_scales[i] / 11)
                layer.ln2.weight.data *= (accum_scales[i] / 11)
    print(f"PID-14.1 Sequential Bootstrapping Complete.")
