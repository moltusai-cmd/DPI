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
    """
    Collects real-time activations after layer_idx (-1 indicates post-embedding).
    Used for sequential manifold analysis to prevent circularity.
    """
    model.eval()
    activations = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Handle standard DataLoader formats [x, y] or [x]
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0].to(device)
            else:
                x_batch = batch.to(device)
                
            # Forward pass up to target layer
            x = model.embedding(x_batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            
            for j in range(layer_idx + 1):
                x = model.layers[j](x)
                
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= num_samples:
                break 
                
    return torch.cat(activations, dim=0)

def get_dct_weights(out_dims, in_dims, warp=1.4):
    """Generates a Zipfian-warped Discrete Cosine Transform (DCT) basis."""
    i = torch.arange(out_dims).view(-1, 1)
    j = torch.arange(in_dims).view(1, -1)
    warped_j = torch.pow(j / in_dims, warp) * in_dims
    W = torch.cos(math.pi / in_dims * (warped_j + 0.5) * i)
    return W * math.sqrt(2.0 / in_dims)

def init_phase0_embedding(model, dataloader, k_samples=2000):
    """
    Phase 0: Lexical Seeding.
    Initializes embeddings via SVD of a Nyström-approximated co-occurrence matrix.
    """
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    k = min(k_samples, vocab_size)
    
    # Construction of a sparse co-occurrence slice
    C_nk = torch.zeros(vocab_size, k, device='cpu')
    print(f"  [Phase 0] Seeding Lexical Manifold (Nyström SVD)...")
    
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        u = x[:, :-1].reshape(-1)
        v = x[:, 1:].reshape(-1)
        
        mask = (v < k)
        u_m, v_m = u[mask], v[mask]
        
        if u_m.numel() > 0:
            C_nk.index_put_((u_m, v_m), torch.ones_like(u_m, dtype=torch.float), accumulate=True)
        if i >= 100: # Sufficient for structural priors
            break
            
    # SVD approximation
    W = C_nk[:k, :].float()
    Uw, Sw, Vw = torch.svd(W)
    inv_Sw = torch.diag(1.0 / torch.sqrt(Sw + 1e-6))
    U_approx = torch.matmul(C_nk.float(), torch.matmul(Uw, inv_Sw))
    
    # Load into embedding weights
    model.embedding.weight.data[:, :min(d_model, k)] = U_approx[:, :min(d_model, k)].to(model.embedding.weight.device)

def initialize_dpi(model, dataloader, warp_zeta=1.0, spectral_gamma=0.25, morph_alpha=0.45, use_calibration=True):
    """
    Main entry point for DPI-14.1 Sequential Bootstrapping.
    
    Args:
        model: The Transformer model instance.
        dataloader: A DataLoader providing sample text for manifold analysis.
        warp_zeta: Zipfian spectral warp factor (default 1.0).
        spectral_gamma: Power-law compression factor (default 0.25).
        morph_alpha: Morphological blending factor (default 0.45).
        use_calibration: Whether to apply Phase 6 LayerNorm calibration.
    """
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    
    # 1. Phase 0: Lexical Seed
    init_phase0_embedding(model, dataloader)
    
    # 2. Global Proto-Clusters
    print("  [Phase 1] Computing Global Lexical Centers...")
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    # 3. Sequential Manifold Sculpting (The DPI-14.1 Core)
    print(f"  [Phase 2] Sequential Bootstrapping of {n_layers} Layers...")
    dct_cache = {}
    
    for l in range(n_layers):
        # COLLECT ACTIVATIONS FROM PREVIOUS LAYER (Real signal flow)
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        
        # Spectral SVD of CURRENT manifold
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        cast_factor = (1.0 - 0.5 * math.sin(math.pi * progress))
        current_gamma = spectral_gamma * cast_factor
        
        # Generate semantic basis
        svd_basis = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        
        # --- MLP Pre-conditioning ---
        ws, wk = math.exp(-progress * 4.0), math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        d_mlp = model.layers[l].mlp.W1.out_features
        
        if (d_mlp, model.d_model) not in dct_cache:
            dct_cache[(d_mlp, model.d_model)] = get_dct_weights(d_mlp, model.d_model, warp=warp_zeta).to(device)
        
        # Blend Zipfian spectral filters with semantic manifold
        mlp_init = (ws * dct_cache[(d_mlp, model.d_model)] + wk * svd_basis.repeat(math.ceil(d_mlp/model.d_model), 1)[:d_mlp]) / (ws + wk)
        model.layers[l].mlp.W1.weight.data = mlp_init
        
        # --- Functional QKV Signatures ---
        ortho_peak = math.sin(math.pi * progress)
        M_k = (1-progress) * centers + progress * svd_basis
        Q_k, _ = torch.linalg.qr(M_k.t()) # Key Signature: Structural Orthogonality
        
        model.layers[l].attn.W_k.weight.data = (1-ortho_peak) * M_k + ortho_peak * Q_k.t()
        
        # Value Signature: Manifold Deployment (High compression)
        svd_v = (U.t() * torch.pow(S + 1e-6, current_gamma * 0.4).unsqueeze(1)).to(device)
        model.layers[l].attn.W_v.weight.data = svd_v
        
        # Query Signature: Routing Alignment (Initial alignment with K)
        alignment = 0.6 * (1.0 - progress)
        model.layers[l].attn.W_q.weight.data = alignment * model.layers[l].attn.W_k.weight.data + (1-alignment) * svd_basis
        
        # Output and MLP Projections: Dynamic Isometry (QR)
        Q_o, _ = torch.linalg.qr(torch.randn(model.d_model, model.d_model, device=device))
        model.layers[l].attn.W_o.weight.data = Q_o
        
        Q2, _ = torch.linalg.qr(torch.randn(d_mlp, d_mlp, device=device))
        model.layers[l].mlp.W2.weight.data = Q2[:model.d_model, :]
        
        if l % 5 == 0 or l == n_layers - 1:
            print(f"    Layer {l:2d} Sculpted (Signal Energy: {S[0].item():.2f})")

    if use_calibration: 
        print("  [Phase 3] Final Manifold Calibration (LayerNorm Scaling)...")
        model.eval()
        with torch.no_grad():
            accum_scales = [torch.zeros(1, device=device) for _ in model.layers]
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    x_batch = batch[0].to(device)
                else:
                    x_batch = batch.to(device)
                    
                x = model.embedding(x_batch) * math.sqrt(model.d_model)
                x = model.pos_encoding(x)
                
                for j, layer in enumerate(model.layers):
                    x = layer(x)
                    accum_scales[j] += torch.sqrt(1.0 / (x.var() + 1e-6))
                if i >= 10: break
                
            for i, layer in enumerate(model.layers):
                scale = accum_scales[i] / 11
                layer.ln1.weight.data *= scale
                layer.ln2.weight.data *= scale
                
    print(f"DPI-14.1 Sequential Bootstrapping Complete.")
