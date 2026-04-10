import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from model import PID8Transformer
import math

def get_activations(model, dataloader, layer_idx):
    """Gets activations after layer_idx. If layer_idx is -1, gets embeddings."""
    model.eval()
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            for i in range(layer_idx + 1):
                x = model.layers[i](x)
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= 1000: # We need ~1000 tokens as per spec
                break
    return torch.cat(activations, dim=0)

def init_phase1_dct(model):
    """Phase 1: Layers 1-2 (Syntax Entry) using DCT-II."""
    print("Phase 1: Initializing Layers 1-2 with DCT-II...")
    d_model = model.d_model
    for i in [0, 1]:
        d_mlp = model.layers[i].mlp.W1.out_features
        # W1: (d_mlp, d_model)
        # W1(i, j) = cos[pi/N * (j + 0.5) * i]
        W1 = torch.zeros(d_mlp, d_model)
        for row in range(d_mlp):
            for col in range(d_model):
                W1[row, col] = math.cos(math.pi / d_model * (col + 0.5) * row)
        model.layers[i].mlp.W1.weight.data = W1.to(model.layers[i].mlp.W1.weight.device)

def init_phase2_kmeans(model, dataloader):
    """Phase 2: Layers 3-4 (Topological Emergence) using Mini-Batch K-Means."""
    print("Phase 2: Initializing Layers 3-4 with K-Means centroids...")
    for i in [2, 3]:
        # Get activations from previous layer
        X = get_activations(model, dataloader, i - 1).cpu().numpy()
        
        # Initialize W_q (d_model, d_model)
        kmeans_q = MiniBatchKMeans(n_clusters=model.d_model, n_init=3, batch_size=1024)
        kmeans_q.fit(X)
        model.layers[i].attn.W_q.weight.data = torch.from_numpy(kmeans_q.cluster_centers_).float().to(model.layers[i].attn.W_q.weight.device)
        
        # Initialize W1 (d_mlp, d_model)
        d_mlp = model.layers[i].mlp.W1.out_features
        kmeans_mlp = MiniBatchKMeans(n_clusters=d_mlp, n_init=3, batch_size=1024)
        kmeans_mlp.fit(X)
        model.layers[i].mlp.W1.weight.data = torch.from_numpy(kmeans_mlp.cluster_centers_).float().to(model.layers[i].mlp.W1.weight.device)

def init_phase3_svd(model, dataloader):
    """Phase 3: Layers 5-6 (Semantic Core) using PCA/SVD."""
    print("Phase 3: Initializing Layers 5-6 with PCA/SVD...")
    for i in [4, 5]:
        X = get_activations(model, dataloader, i - 1) # (N, d_model)
        X = X - X.mean(dim=0)
        
        # Covariance C = 1/N * X.T * X
        C = torch.matmul(X.t(), X) / X.size(0)
        U, S, V = torch.svd(C)
        
        # U is (d_model, d_model). We use it to initialize weights.
        # Projecting principal components scaled by expected variance (here S)
        # For simplicity, we use U to initialize W_q, W_k, W_v
        model.layers[i].attn.W_q.weight.data = (U.t() * torch.sqrt(S).unsqueeze(1)).to(model.layers[i].attn.W_q.weight.device)
        model.layers[i].attn.W_k.weight.data = U.t().to(model.layers[i].attn.W_k.weight.device)
        model.layers[i].attn.W_v.weight.data = U.t().to(model.layers[i].attn.W_v.weight.device)
        
        # For MLP W1 (d_mlp, d_model), we might need to pad or repeat U if d_mlp > d_model
        d_mlp = model.layers[i].mlp.W1.out_features
        W1_init = U.t().repeat(math.ceil(d_mlp / model.d_model), 1)[:d_mlp]
        model.layers[i].mlp.W1.weight.data = W1_init.to(model.layers[i].mlp.W1.weight.device)

def init_phase4_qr(model):
    """Phase 4: Layer 7 (Compression) using Orthogonal Matrices (QR)."""
    print("Phase 4: Initializing Layer 7 with Orthogonal Matrices (QR)...")
    i = 6 # Layer 7
    # W_o (d_model, d_model)
    M_o = torch.randn(model.d_model, model.d_model)
    Q_o, R_o = torch.linalg.qr(M_o)
    model.layers[i].attn.W_o.weight.data = Q_o.to(model.layers[i].attn.W_o.weight.device)
    
    # W2 (d_model, d_mlp)
    d_mlp = model.layers[i].mlp.W1.out_features
    # We need an orthogonal projection from d_mlp to d_model
    # We can take a d_mlp x d_mlp orthogonal matrix and truncate
    M_mlp = torch.randn(d_mlp, d_mlp)
    Q_mlp, R_mlp = torch.linalg.qr(M_mlp)
    model.layers[i].mlp.W2.weight.data = Q_mlp[:model.d_model, :].to(model.layers[i].mlp.W2.weight.device)

def init_phase5_whitening(model, dataloader):
    """Phase 5: Layer 8 & Unembedding (Whitening)."""
    print("Phase 5: Initializing Layer 8 and Unembedding with Mahalanobis Whitening...")
    # Get activations after layer 7
    X = get_activations(model, dataloader, 6) # (N, d_model)
    
    mu = X.mean(dim=0)
    X_centered = X - mu
    C = torch.matmul(X_centered.t(), X_centered) / X.size(0)
    
    # Mahalanobis whitening: C^-1/2
    U, S, V = torch.svd(C)
    eps = 1e-5
    whitening_matrix = torch.matmul(U, torch.matmul(torch.diag(1.0 / torch.sqrt(S + eps)), U.t()))
    
    # Apply to unembedding W_unembed (vocab_size, d_model)
    # W_unembed = (W_unembed - directional_mean) * whitening_matrix
    model.unembed.weight.data = torch.matmul(model.unembed.weight.data - mu.to(model.unembed.weight.device), whitening_matrix.to(model.unembed.weight.device))

def init_phase0_embedding(model, dataloader):
    """Phase 0: Initialize Embedding via SVD of Co-occurrence Matrix."""
    print("Phase 0: Seeding Embeddings with Co-occurrence SVD...")
    vocab_size = model.embedding.num_embeddings
    d_model = model.d_model
    
    # Simple co-occurrence (neighboring tokens)
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
        if count > 50: break # Use small sample
    
    # SVD for dimensionality reduction (Vocab -> d_model)
    # We use a sparse-friendly approach or just small subset for speed
    # For 16k vocab, dense SVD is okay but slow. Let's use a simpler spectral init.
    # We'll use the covariance of the co-occurrence counts.
    U, S, V = torch.svd(cooc[:2000, :2000].float()) # Spectral seed on top 2000 tokens
    seed = torch.randn(vocab_size, d_model) * 0.02
    seed[:2000, :min(d_model, 2000)] = U[:, :min(d_model, 2000)]
    model.embedding.weight.data = seed.to(model.embedding.weight.device)

def init_phase6_calibration(model, dataloader):
    """Phase 6: Calibrate LayerNorms to ensure unit variance in residual stream."""
    print("Phase 6: Calibrating Residual Variance...")
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = model.embedding(batch) * math.sqrt(model.d_model)
            x = model.pos_encoding(x)
            
            for i, layer in enumerate(model.layers):
                # Measure variance before layer
                var_in = x.var()
                
                # Forward through layer
                x = layer(x)
                
                # Measure variance after
                var_out = x.var()
                
                # Adjust LayerNorm gain to stabilize
                # scale = sqrt(1 / var_out)
                scale = torch.sqrt(1.0 / (var_out + 1e-6))
                layer.ln1.weight.data *= scale
                layer.ln2.weight.data *= scale
            break # One batch is enough for calibration

def initialize_pid8(model, dataloader):
    init_phase0_embedding(model, dataloader)
    init_phase1_dct(model)
    init_phase2_kmeans(model, dataloader)
    init_phase3_svd(model, dataloader)
    init_phase4_qr(model)
    init_phase5_whitening(model, dataloader)
    init_phase6_calibration(model, dataloader)
    print("PID-8 (Fertile Soil Edition) Initialization Complete.")

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create model
    model = PID8Transformer()
    
    # Create dummy dataloader
    dummy_data = torch.randint(0, 16384, (10, 512)) # 10 samples, 512 tokens
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Helper to wrap dataloader to return only X
    class SimpleLoader:
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for batch in self.dl: yield batch[0]
            
    initialize_pid8(model, SimpleLoader(dataloader))
    
    # Basic verification
    print("\nDetailed Verification:")
    
    # Phase 1: Check DCT W1 values
    W1_l1 = model.layers[0].mlp.W1.weight.data
    expected_W1_0_0 = math.cos(math.pi / model.d_model * 0.5 * 0)
    print(f"Phase 1 - Layer 1 W1[0,0]: {W1_l1[0,0]:.4f} (Expected: {expected_W1_0_0:.4f})")
    
    # Phase 4: Check Layer 7 W_o orthogonality
    Wo = model.layers[6].attn.W_o.weight.data
    ortho_err = torch.norm(torch.matmul(Wo, Wo.t()) - torch.eye(model.d_model))
    print(f"Phase 4 - Layer 7 W_o Orthogonality Error: {ortho_err:.4f}")
    
    # Phase 5: Check Unembedding whitening
    # If correctly whitened, the covariance of (unembed.weight.data) should be I
    W_unembed = model.unembed.weight.data
    W_centered = W_unembed - W_unembed.mean(dim=0)
    C_unembed = torch.matmul(W_centered.t(), W_centered) / W_unembed.size(0)
    # Actually, the whitening is based on the activations, not the weights themselves
    # But we can check the weight magnitude
    print(f"Phase 5 - Unembedding weight mean: {W_unembed.mean():.6f}")
    
    # Check parameter count again
    from model import count_parameters
    print(f"Total Parameters: {count_parameters(model) / 1e6:.2f}M")
