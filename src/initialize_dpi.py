import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v17.0 - "Platonic Synchronizer" Edition
Universal Geometry invariant to stochastic seeds.
"""

def canonicalize_svd(U, S, V):
    """Forces consistent signs for SVD components based on max absolute value."""
    # U: (N, K), V: (M, K)
    max_abs_cols = torch.argmax(torch.abs(U), dim=0)
    signs = torch.sign(U[max_abs_cols, torch.arange(U.shape[1])])
    U *= signs
    V *= signs
    return U, S, V

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

def spectral_normalize(W, target_sigma=1.0):
    if W.dim() < 2: return W
    U, S, V = torch.svd_lowrank(W.flatten(1), q=1)
    sigma = S[0]
    if sigma > 1e-8:
        return W * (target_sigma / sigma)
    return W

def init_phase0_lexical(model, dataloader):
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    device = next(model.parameters()).device
    print(f"  [Phase 0] Seeding Lexical Manifold (Spectral Isometry)...")
    C = torch.zeros((vocab_size, vocab_size), device=device)
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)): x = batch[0].to(device)
        else: x = batch.to(device)
        u, v = x[:, :-1].reshape(-1), x[:, 1:].reshape(-1)
        indices = u * vocab_size + v
        C.view(-1).index_add_(0, indices, torch.ones_like(u, dtype=torch.float, device=device))
        if i >= 200: break
    
    # Canonical SVD for Phase 0
    U, S, V = torch.svd_lowrank(C.float(), q=d_model, niter=10)
    U, S, V = canonicalize_svd(U, S, V)
    
    model.embedding.weight.data[:, :min(d_model, vocab_size)] = U[:, :min(d_model, vocab_size)]
    model.embedding.weight.data = spectral_normalize(model.embedding.weight.data, target_sigma=1.0)
    return U, V

def initialize_dpi(model, dataloader, spectral_gamma=0.25, use_calibration=True, mlp_jitter=0.01, mode="v17.0"):
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    phase_shift_layer = int(0.42 * n_layers)
    
    # 1. Phase 0: Lexical
    U_lex, V_lex = init_phase0_lexical(model, dataloader)
    
    # 2. Phase 1: Semantic Clustering (Synchronized)
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    # Fixed seed for KMeans internal stochasticity, but centers will be sorted
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=5, batch_size=1024, random_state=42).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    # Sort centers by L2 norm to ensure consistent cluster IDs across runs
    center_norms = torch.norm(centers, dim=1)
    _, sort_idx = torch.sort(center_norms, descending=True)
    centers = centers[sort_idx]
    
    print(f"  [Phase 2] Genomic Bootstrapping: {mode.upper()} (Platonic Synchronizer)...")
    
    for l in range(n_layers):
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        # Canonical SVD for each layer bootstrapping
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        U, S, V = canonicalize_svd(U, S, V)
        
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        current_gamma = spectral_gamma * (1.0 - 0.25 * math.log(1.0 + progress))
        svd_basis = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        
        layer = model.layers[l]
        attn = getattr(layer, 'attn', None) or getattr(layer, 'attention', None)
        mlp = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        
        # MLP Init
        W1 = getattr(mlp, 'W1', None) or getattr(mlp, 'fc1', None)
        W_gate = getattr(mlp, 'W_gate', None)
        W2 = getattr(mlp, 'W2', None) or getattr(mlp, 'fc2', None)
        mlp_basis = svd_basis.repeat(W1.out_features // model.d_model, 1)
        W1.weight.data = spectral_normalize(mlp_basis + torch.randn_like(mlp_basis) * mlp_jitter, target_sigma=1.0)
        if W_gate is not None:
            W_gate.weight.data = spectral_normalize(mlp_basis + torch.randn_like(mlp_basis) * mlp_jitter, target_sigma=1.0)
        W2.weight.data = spectral_normalize(svd_basis.t().repeat(1, W1.out_features // model.d_model), target_sigma=res_scale)

        # Attention Init
        W_q = getattr(attn, 'W_q', None) or getattr(attn, 'q_proj', None)
        W_k = getattr(attn, 'W_k', None) or getattr(attn, 'k_proj', None)
        W_v = getattr(attn, 'W_v', None) or getattr(attn, 'v_proj', None)
        W_o = getattr(attn, 'W_o', None) or getattr(attn, 'o_proj', None)
        
        n_heads = attn.n_heads
        d_head = model.d_model // n_heads
        is_consolidated = (l >= phase_shift_layer)
        
        W_q_data = W_q.weight.data.view(n_heads, d_head, model.d_model)
        W_k_data = W_k.weight.data.view(n_heads, d_head, model.d_model)
        W_v_data = W_v.weight.data.view(n_heads, d_head, model.d_model)
        
        for h_idx in range(n_heads):
            align = 0.0001 if is_consolidated else [0.80, 0.40, 0.05][min(2, int(3 * h_idx / n_heads))]
            if not is_consolidated:
                W_k_data[h_idx] = spectral_normalize(centers[h_idx*d_head : (h_idx+1)*d_head] + 0.1 * svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0)
                W_v_data[h_idx] = spectral_normalize(svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0)
                W_q_data[h_idx] = spectral_normalize(align * W_k_data[h_idx] + (1 - align) * svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0)
            else:
                W_k_data[h_idx] = spectral_normalize(svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0)
                W_v_data[h_idx] = W_k_data[h_idx].clone()
                # Orthogonal Query Bootstrap (Synchronized)
                Q_rand = torch.ones_like(W_k_data[h_idx]) # Constant seed for ortho-basis
                dot = (Q_rand * W_k_data[h_idx]).sum(dim=1, keepdim=True)
                norm_k = (W_k_data[h_idx] * W_k_data[h_idx]).sum(dim=1, keepdim=True)
                Q_ortho = Q_rand - (dot / (norm_k + 1e-8)) * W_k_data[h_idx]
                W_q_data[h_idx] = spectral_normalize(align * W_k_data[h_idx] + (1 - align) * Q_ortho, target_sigma=1.0)
        
        W_o.weight.data = spectral_normalize(svd_basis.t(), target_sigma=res_scale)

    # 4. Phase 4: LM Head
    unembed = getattr(model, 'unembed', None) or getattr(model, 'lm_head', None)
    if unembed:
        print(f"  [Phase 4] Calibrating Output Head (Spectral Stability)...")
        v_h, v_w = V_lex.shape
        u_h, u_w = unembed.weight.data.shape
        h, w = min(v_h, u_h), min(v_w, u_w)
        unembed.weight.data[:h, :w] = V_lex[:h, :w]
        unembed.weight.data = spectral_normalize(unembed.weight.data, target_sigma=1.0)

    # Capture anchors for DSO
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.dpi_anchor = p.data.clone().detach()
            p.dpi_layer_idx = int(name.split('layers.')[1].split('.')[0]) if 'layers.' in name else -1

    print(f"DPI-V17.0 PLATONIC GOLD STANDARD COMPLETE.")
    return {}
