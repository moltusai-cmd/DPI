import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from model import PID8Transformer
import math

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
            if len(activations) * x.size(1) >= 1000:
                break
    return torch.cat(activations, dim=0)

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

def init_phase1_dct(model, zipf_warp=1.0):
    """Phase 1: DCT with Zipfian Warping."""
    d_model = model.d_model
    for i in [0, 1]:
        d_mlp = model.layers[i].mlp.W1.out_features
        W1 = torch.zeros(d_mlp, d_model)
        for row in range(d_mlp):
            for col in range(d_model):
                # Warping the frequency index
                warped_col = math.pow(col / d_model, zipf_warp) * d_model
                W1[row, col] = math.cos(math.pi / d_model * (warped_col + 0.5) * row)
        model.layers[i].mlp.W1.weight.data = W1.to(model.layers[i].mlp.W1.weight.device)

def init_phase2_kmeans(model, dataloader):
    for i in [2, 3]:
        X = get_activations(model, dataloader, i - 1).cpu().numpy()
        kmeans_q = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024)
        kmeans_q.fit(X)
        model.layers[i].attn.W_q.weight.data = torch.from_numpy(kmeans_q.cluster_centers_).float().to(model.layers[i].attn.W_q.weight.device)
        d_mlp = model.layers[i].mlp.W1.out_features
        kmeans_mlp = MiniBatchKMeans(n_clusters=d_mlp, n_init=3, batch_size=1024)
        kmeans_mlp.fit(X)
        model.layers[i].mlp.W1.weight.data = torch.from_numpy(kmeans_mlp.cluster_centers_).float().to(model.layers[i].mlp.W1.weight.device)

def init_phase3_svd(model, dataloader, spectral_gamma=0.5):
    """Phase 3: SVD with Spectral Gamma power."""
    for i in [4, 5]:
        X = get_activations(model, dataloader, i - 1)
        X = X - X.mean(dim=0)
        C = torch.matmul(X.t(), X) / X.size(0)
        U, S, V = torch.svd(C)
        model.layers[i].attn.W_q.weight.data = (U.t() * torch.pow(S + 1e-6, spectral_gamma).unsqueeze(1)).to(model.layers[i].attn.W_q.weight.device)
        model.layers[i].attn.W_k.weight.data = U.t().to(model.layers[i].attn.W_k.weight.device)
        model.layers[i].attn.W_v.weight.data = U.t().to(model.layers[i].attn.W_v.weight.device)
        d_mlp = model.layers[i].mlp.W1.out_features
        W1_init = U.t().repeat(math.ceil(d_mlp / model.d_model), 1)[:d_mlp]
        model.layers[i].mlp.W1.weight.data = W1_init.to(model.layers[i].mlp.W1.weight.device)

def init_phase4_qr(model):
    for i in [6]:
        M_o = torch.randn(model.d_model, model.d_model)
        Q_o, R_o = torch.linalg.qr(M_o)
        model.layers[i].attn.W_o.weight.data = Q_o.to(model.layers[i].attn.W_o.weight.device)
        d_mlp = model.layers[i].mlp.W1.out_features
        M_mlp = torch.randn(d_mlp, d_mlp)
        Q_mlp, R_mlp = torch.linalg.qr(M_mlp)
        model.layers[i].mlp.W2.weight.data = Q_mlp[:model.d_model, :].to(model.layers[i].mlp.W2.weight.device)

def init_phase5_whitening(model, dataloader):
    X = get_activations(model, dataloader, 6)
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

def init_phase7_morphing(model, morph_alpha=0.2):
    with torch.no_grad():
        for i in range(1, len(model.layers)):
            if model.layers[i].mlp.W1.weight.data.shape == model.layers[i-1].mlp.W1.weight.data.shape:
                model.layers[i].mlp.W1.weight.data = (1 - morph_alpha) * model.layers[i].mlp.W1.weight.data + morph_alpha * model.layers[i-1].mlp.W1.weight.data
            if model.layers[i].attn.W_q.weight.data.shape == model.layers[i-1].attn.W_q.weight.data.shape:
                model.layers[i].attn.W_q.weight.data = (1 - morph_alpha) * model.layers[i].attn.W_q.weight.data + morph_alpha * model.layers[i-1].attn.W_q.weight.data

def initialize_pid8(model, dataloader, zipf_warp=1.4, spectral_gamma=0.35, morph_alpha=0.35):
    init_phase0_embedding(model, dataloader)
    init_phase1_dct(model, zipf_warp=zipf_warp)
    init_phase2_kmeans(model, dataloader)
    init_phase3_svd(model, dataloader, spectral_gamma=spectral_gamma)
    init_phase4_qr(model)
    init_phase5_whitening(model, dataloader)
    init_phase7_morphing(model, morph_alpha=morph_alpha)
    init_phase6_calibration(model, dataloader)
