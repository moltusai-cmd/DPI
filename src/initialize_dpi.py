import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v16.6 - Concentrated Isometry + Residual Stability
"""

def zipf_spectral_warp(dim, zeta=0.1, device='cpu'):
    """
    Generates a Zipfian (Power Law) spectrum for singular values.
    s_i = i^(-zeta)
    """
    indices = torch.arange(1, dim + 1, device=device).float()
    spectrum = torch.pow(indices, -zeta)
    return spectrum / spectrum[0] # Normalize max to 1.0

def heavy_tail_normalize(W, zeta=0.1, target_sigma=1.0):
    """
    Injects a Zipfian Heavy-Tail spectrum into the weight matrix.
    Ensures rank is preserved but singular values follow a power law.
    """
    if W.dim() < 2: return W
    device = W.device
    flat_W = W.flatten(1)
    rows, cols = flat_W.shape
    q = min(rows, cols)
    U, S, V = torch.linalg.svd(flat_W, full_matrices=False)
    new_S = zipf_spectral_warp(len(S), zeta=zeta, device=device) * target_sigma
    W_new = torch.matmul(U * torch.diag(new_S), V)
    return W_new.view_as(W)

def concentrated_isometry_normalize(W, epsilon=0.05, target_sigma=1.0):
    """
    DPI v16.6: Concentrated Isometry.
    Injects a small structured perturbation (eta) around the identity spectrum.
    S_ii = 1 + epsilon * eta_i
    """
    if W.dim() < 2: return W
    device = W.device
    flat_W = W.flatten(1)
    U, S, V = torch.linalg.svd(flat_W, full_matrices=False)
    if len(S) > 1:
        eta = (S - S.mean()) / (S.std() + 1e-8)
        eta = torch.tanh(eta)
    else:
        eta = torch.zeros_like(S)
    new_S = (1.0 + epsilon * eta) * target_sigma
    W_new = torch.matmul(U * torch.diag(new_S), V)
    return W_new.view_as(W)

def spectral_normalize(W, target_sigma=1.0, mode="v16.6", epsilon=0.05, zeta=0.1):
    if mode == "v16.6": return concentrated_isometry_normalize(W, epsilon=epsilon, target_sigma=target_sigma)
    if mode == "v16.5": return heavy_tail_normalize(W, zeta=zeta, target_sigma=target_sigma)
    if W.dim() < 2: return W
    U, S, V = torch.svd_lowrank(W.flatten(1), q=1)
    sigma = S[0]
    if sigma > 1e-8: return W * (target_sigma / sigma)
    return W

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

def init_phase0_lexical(model, dataloader, mode="v16.6", epsilon=0.05, zeta=0.1):
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
    U, S, V = torch.svd_lowrank(C.float(), q=d_model, niter=10)
    model.embedding.weight.data[:, :min(d_model, vocab_size)] = U[:, :min(d_model, vocab_size)]
    model.embedding.weight.data = spectral_normalize(model.embedding.weight.data, target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
    return U, V

def initialize_dpi(model, dataloader, spectral_gamma=0.25, use_calibration=True, mlp_jitter=0.02, mode="v16.6", epsilon=0.05, zeta=0.1):
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    phase_shift_layer = int(0.42 * n_layers)
    spectral_map = {}
    U_lex, V_lex = init_phase0_lexical(model, dataloader, mode=mode, epsilon=epsilon, zeta=zeta)
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    print(f"  [Phase 2] Genomic Bootstrapping: {mode.upper()} (Concentrated Isometry)...")
    for l in range(n_layers):
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        current_gamma = spectral_gamma * (1.0 - 0.25 * math.log(1.0 + progress))
        spectral_map[l] = current_gamma
        svd_basis = (U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device)
        layer = model.layers[l]
        attn = getattr(layer, 'attn', None) or getattr(layer, 'attention', None)
        mlp = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        W1 = getattr(mlp, 'W1', None) or getattr(mlp, 'fc1', None)
        W_gate = getattr(mlp, 'W_gate', None)
        W2 = getattr(mlp, 'W2', None) or getattr(mlp, 'fc2', None)
        mlp_basis = svd_basis.repeat(W1.out_features // model.d_model, 1)
        W1.weight.data = spectral_normalize(mlp_basis + torch.randn_like(mlp_basis) * mlp_jitter, target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
        if W_gate is not None:
            W_gate.weight.data = spectral_normalize(mlp_basis + torch.randn_like(mlp_basis) * mlp_jitter, target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
        W2.weight.data = spectral_normalize(svd_basis.t().repeat(1, W1.out_features // model.d_model), target_sigma=res_scale, mode=mode, epsilon=epsilon, zeta=zeta)
        W_q = getattr(attn, 'W_q', None) or getattr(attn, 'q_proj', None)
        W_k = getattr(attn, 'W_k', None) or getattr(attn, 'k_proj', None)
        W_v = getattr(attn, 'W_v', None) or getattr(attn, 'v_proj', None)
        W_o = getattr(attn, 'W_o', None) or getattr(attn, 'o_proj', None)
        n_heads = attn.n_heads
        d_head = model.d_model // n_heads
        is_consolidated = (l >= phase_shift_layer)
        alignments = []
        for h_idx in range(n_heads):
            head_progress = h_idx / n_heads
            if is_consolidated: alignments.append(0.0001)
            else:
                if head_progress < 0.25: alignments.append(0.80)
                elif head_progress < 0.75: alignments.append(0.40)
                else: alignments.append(0.05)
        W_q_data = W_q.weight.data.view(n_heads, d_head, model.d_model)
        W_k_data = W_k.weight.data.view(n_heads, d_head, model.d_model)
        W_v_data = W_v.weight.data.view(n_heads, d_head, model.d_model)
        for h_idx in range(n_heads):
            align = alignments[h_idx]
            if not is_consolidated:
                W_k_data[h_idx] = spectral_normalize(centers[h_idx*d_head : (h_idx+1)*d_head] + 0.2 * svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
                W_v_data[h_idx] = spectral_normalize(svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
                W_q_data[h_idx] = spectral_normalize(align * W_k_data[h_idx] + (1 - align) * svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
            else:
                W_k_data[h_idx] = spectral_normalize(svd_basis[h_idx*d_head : (h_idx+1)*d_head], target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
                W_v_data[h_idx] = W_k_data[h_idx].clone()
                Q_rand = torch.randn_like(W_k_data[h_idx])
                dot = (Q_rand * W_k_data[h_idx]).sum(dim=1, keepdim=True)
                norm_k = (W_k_data[h_idx] * W_k_data[h_idx]).sum(dim=1, keepdim=True)
                Q_ortho = Q_rand - (dot / (norm_k + 1e-8)) * W_k_data[h_idx]
                W_q_data[h_idx] = spectral_normalize(align * W_k_data[h_idx] + (1 - align) * Q_ortho, target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
        W_o.weight.data = spectral_normalize(svd_basis.t(), target_sigma=res_scale, mode=mode, epsilon=epsilon, zeta=zeta)
    if use_calibration:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)): x = batch[0].to(device)
                else: x = batch.to(device)
                x = model.embedding(x); x = model.pos_encoding(x)
                for j, layer in enumerate(model.layers):
                    x = layer(x)
                    target_var = 1.0 + 0.5 * math.sin(math.pi * (j / n_layers))
                    scale = torch.sqrt(target_var / (x.var() + 1e-6))
                    layer.ln1.weight.data *= scale; layer.ln2.weight.data *= scale
                if i >= 5: break
    unembed = getattr(model, 'unembed', None) or getattr(model, 'lm_head', None)
    if unembed:
        print(f"  [Phase 4] Calibrating Output Head (Spectral Stability)...")
        v_h, v_w = V_lex.shape
        u_h, u_w = unembed.weight.data.shape
        h, w = min(v_h, u_h), min(v_w, u_w)
        unembed.weight.data[:h, :w] = V_lex[:h, :w]
        unembed.weight.data = spectral_normalize(unembed.weight.data, target_sigma=1.0, mode=mode, epsilon=epsilon, zeta=zeta)
    print(f"DPI-{mode.upper()} GOLD STANDARD COMPLETE (Concentrated Isometry).")
    return spectral_map
