import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v15.1 - Genomic Jitter & Asymmetric Scaling
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

def get_dct_weights(out_dims, in_dims, warp=1.4):
    i = torch.arange(out_dims).view(-1, 1).float()
    j = torch.arange(in_dims).view(1, -1).float()
    warped_j = torch.pow(j / in_dims, warp) * in_dims
    W = torch.cos(math.pi / in_dims * (warped_j + 0.5) * i)
    return W * math.sqrt(2.0 / in_dims)

def normalize_weight(W, target_std=None):
    if target_std is None: target_std = math.sqrt(1.0 / W.size(1))
    curr_std = W.std()
    if curr_std > 1e-8: return W * (target_std / curr_std)
    return W

def init_phase0_embedding(model, dataloader, use_exact_svd=True):
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

def initialize_dpi(model, dataloader, warp_zeta=1.1, spectral_gamma=0.25, use_calibration=True, use_exact_svd=True, residual_scale=1.0, mlp_jitter=0.02, gamma_dict=None, jitter_dict=None):
    """
    Args:
        jitter_dict: Optional dict for genomic jitter levels: {'q': 0.04, 'k': 0.01, 'v': 0.015, 'o': 0.035, 'mlp_w1': 0.02, 'mlp_w2': 0.04}
    """
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    init_phase0_embedding(model, dataloader, use_exact_svd=use_exact_svd)
    
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, model.d_model))
    km = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024).fit(X_lex.cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    # Defaults
    if gamma_dict is None: gamma_dict = {'q': spectral_gamma, 'k': spectral_gamma, 'v': spectral_gamma, 'mlp': spectral_gamma}
    if jitter_dict is None: jitter_dict = {'q': mlp_jitter, 'k': mlp_jitter, 'v': mlp_jitter, 'o': mlp_jitter, 'mlp_w1': mlp_jitter, 'mlp_w2': mlp_jitter}
    
    print(f"  [Phase 2] Sequential Bootstrapping v15.1 (Genomic Jitter)...")
    dct_cache = {}
    
    for l in range(n_layers):
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, model.d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        U, S, V = torch.svd(torch.matmul(X_centered.t(), X_centered) / X_centered.size(0))
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        
        # --- Differentiated Gammas ---
        def get_g(key):
            val = gamma_dict.get(key, spectral_gamma)
            return val[0] + (val[1] - val[0]) * progress if isinstance(val, list) else val
        g_q, g_k, g_v, g_mlp = get_g('q'), get_g('k'), get_g('v'), get_g('mlp')
        
        layer = model.layers[l]
        attn = getattr(layer, 'attn', None) or getattr(layer, 'attention', None)
        mlp = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
        
        # 1. MLP Init
        W1 = getattr(mlp, 'W1', None) or getattr(mlp, 'fc1', None); W2 = getattr(mlp, 'W2', None) or getattr(mlp, 'fc2', None)
        d_mlp = W1.out_features
        if (d_mlp, model.d_model) not in dct_cache: dct_cache[(d_mlp, model.d_model)] = get_dct_weights(d_mlp, model.d_model, warp=warp_zeta).to(device)
        svd_mlp = normalize_weight((U.t() * torch.pow(S + 1e-6, g_mlp).unsqueeze(1)).to(device))
        ws, wk = math.exp(-progress * 4.0), math.exp(-0.5 * ((progress - 0.5) / 0.25)**2)
        w1_init = (ws * dct_cache[(d_mlp, model.d_model)] + wk * svd_mlp.repeat(math.ceil(d_mlp/model.d_model), 1)[:d_mlp]) / (ws + wk)
        # Apply Component-Specific Jitter
        if jitter_dict['mlp_w1'] > 0: w1_init += torch.randn_like(w1_init) * jitter_dict['mlp_w1']
        W1.weight.data = normalize_weight(w1_init)
        
        # 2. Attention Init
        W_q = getattr(attn, 'W_q', None) or getattr(attn, 'q_proj', None); W_k = getattr(attn, 'W_k', None) or getattr(attn, 'k_proj', None)
        W_v = getattr(attn, 'W_v', None) or getattr(attn, 'v_proj', None); W_o = getattr(attn, 'W_o', None) or getattr(attn, 'o_proj', None)
        
        # K (Genomic Jitter)
        svd_k = normalize_weight((U.t() * torch.pow(S + 1e-6, g_k).unsqueeze(1)).to(device))
        ortho_peak = math.sin(math.pi * progress); M_k = (1-progress) * centers + progress * svd_k; Q_k, _ = torch.linalg.qr(M_k.t())
        wk_init = (1-ortho_peak) * M_k + ortho_peak * Q_k.t()
        if jitter_dict['k'] > 0: wk_init += torch.randn_like(wk_init) * jitter_dict['k']
        W_k.weight.data = normalize_weight(wk_init)
        
        # V (Genomic Jitter)
        svd_v = normalize_weight((U.t() * torch.pow(S + 1e-6, g_v).unsqueeze(1)).to(device))
        if jitter_dict['v'] > 0: svd_v += torch.randn_like(svd_v) * jitter_dict['v']
        W_v.weight.data = normalize_weight(svd_v)
        
        # Q (Genomic Jitter - Highest)
        svd_q = normalize_weight((U.t() * torch.pow(S + 1e-6, g_q).unsqueeze(1)).to(device))
        alignment = 0.6 * (1.0 - progress); wq_init = alignment * W_k.weight.data + (1-alignment) * svd_q
        if jitter_dict['q'] > 0: wq_init += torch.randn_like(wq_init) * jitter_dict['q']
        W_q.weight.data = normalize_weight(wq_init)
        
        # 3. Residual Stability & O-Proj Jitter
        res_scale = residual_scale / math.sqrt(2 * n_layers) if residual_scale == 1.0 else residual_scale
        Q_o, _ = torch.linalg.qr(torch.randn(model.d_model, model.d_model, device=device))
        if jitter_dict['o'] > 0: Q_o += torch.randn_like(Q_o) * jitter_dict['o']
        W_o.weight.data = normalize_weight(Q_o, target_std=res_scale * math.sqrt(1.0/model.d_model))
        
        w2_init = torch.linalg.qr(torch.randn(d_mlp, d_mlp, device=device))[0][:model.d_model, :]
        if jitter_dict['mlp_w2'] > 0: w2_init += torch.randn_like(w2_init) * jitter_dict['mlp_w2']
        W2.weight.data = normalize_weight(w2_init, target_std=res_scale * math.sqrt(1.0/d_mlp))
        
        if l % 5 == 0 or l == n_layers - 1:
            print(f"    Layer {l:2d} | Genomic Jitters applied.")

    if use_calibration: 
        print("  [Phase 3] Final Calibration...")
        model.eval()
        with torch.no_grad():
            accum_scales = [torch.zeros(1, device=device) for _ in model.layers]
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)): x_batch = batch[0].to(device)
                else: x_batch = batch.to(device)
                x = model.embedding(x_batch); x = model.pos_encoding(x)
                for j, layer in enumerate(model.layers): x = layer(x); accum_scales[j] += torch.sqrt(1.0 / (x.var() + 1e-6))
                if i >= 10: break
            for i, layer in enumerate(model.layers):
                scale = torch.clamp(accum_scales[i] / 11, 0.1, 2.0)
                getattr(layer, 'ln1').weight.data *= scale; getattr(layer, 'ln2').weight.data *= scale
    print(f"DPI-15.1 Genomic Initialization Complete.")
