import torch
import torch.nn as nn
import sys
import os
import numpy as np
import json
import math

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import PID8Transformer

def get_stats(W):
    w_np = W.detach().cpu().numpy().flatten()
    return {
        "mean": float(np.mean(w_np)),
        "std": float(np.std(w_np)),
        "kurtosis": float((np.mean((w_np - np.mean(w_np))**4) / (np.std(w_np)**4 + 1e-10)) - 3),
        "norm": float(torch.norm(W).item())
    }

def get_spectral_data(W):
    # Ensure 2D
    if W.dim() > 2: W = W.view(W.size(0), -1)
    S = torch.linalg.svdvals(W.float())
    s_np = S.cpu().numpy()
    
    # Power-law Alpha (tail decay)
    log_indices = np.log(np.arange(1, len(s_np) + 1))
    log_S = np.log(s_np + 1e-10)
    # Fit on the middle-to-end section of the spectrum
    start_idx = int(len(s_np) * 0.1)
    slope, _ = np.polyfit(log_indices[start_idx:], log_S[start_idx:], 1)
    
    # Effective Rank (Shannon Entropy of spectrum)
    p = s_np / (np.sum(s_np) + 1e-10)
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-10)))
    
    return {"alpha": float(-slope), "eff_rank": float(eff_rank)}

def get_harmonics(W, top_k=3):
    if W.dim() > 2: W = W.view(W.size(0), -1)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    # Analyze the primary singular vector (The "Genome sequence")
    U0 = U[:, 0].cpu().numpy()
    fft_vals = np.abs(np.fft.rfft(U0))
    freqs = np.fft.rfftfreq(len(U0))
    # Ignore DC component
    idx = np.argsort(fft_vals[1:])[::-1][:top_k] + 1
    return [{"freq": float(freqs[i]), "power": float(fft_vals[i])} for i in idx]

def get_ortho_error(W):
    if W.dim() > 2: W = W.view(W.size(0), -1)
    m, n = W.shape
    device = W.device
    if m <= n:
        I = torch.eye(m, device=device)
        err = torch.norm(torch.mm(W, W.T) - I) / m
    else:
        I = torch.eye(n, device=device)
        err = torch.norm(torch.mm(W.T, W) - I) / n
    return float(err.item())

def sequence_genome(model_path, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = dict(vocab_size=16384, d_model=320, n_heads=5, d_mlp=1280, n_layers=8)
    model = PID8Transformer(**params).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    genome = {
        "metadata": {"path": model_path, "layers": params["n_layers"]},
        "embedding": {},
        "layers": [],
        "unembed": {}
    }

    print(f"Sequencing Genome: {model_path}...")

    # 1. Embedding
    ew = model.embedding.weight.data
    genome["embedding"] = {
        "stats": get_stats(ew),
        "spectral": get_spectral_data(ew),
        "harmonics": get_harmonics(ew)
    }

    # 2. Layers
    for i in range(params["n_layers"]):
        l = model.layers[i]
        layer_data = {"index": i, "components": {}}
        
        # Attention
        for name, W in [("q", l.attn.W_q.weight.data), 
                        ("k", l.attn.W_k.weight.data), 
                        ("v", l.attn.W_v.weight.data), 
                        ("o", l.attn.W_o.weight.data)]:
            layer_data["components"][name] = {
                "spectral": get_spectral_data(W),
                "stats": get_stats(W),
                "ortho_error": get_ortho_error(W),
                "harmonics": get_harmonics(W)
            }
        
        # QK Alignment (Precise Dot Product)
        wq = l.attn.W_q.weight.data
        wk = l.attn.W_k.weight.data
        q_norm = torch.nn.functional.normalize(wq.view(-1), dim=0)
        k_norm = torch.nn.functional.normalize(wk.view(-1), dim=0)
        layer_data["components"]["qk_alignment"] = float(torch.dot(q_norm, k_norm).item())

        # MLP
        w1 = l.mlp.W1.weight.data
        w2 = l.mlp.W2.weight.data
        layer_data["components"]["w1"] = {"spectral": get_spectral_data(w1), "stats": get_stats(w1)}
        layer_data["components"]["w2"] = {"spectral": get_spectral_data(w2), "stats": get_stats(w2)}
        
        # MLP Effective (W2 * W1)
        m_eff = torch.mm(w2, w1)
        layer_data["components"]["mlp_effective"] = {
            "spectral": get_spectral_data(m_eff),
            "harmonics": get_harmonics(m_eff)
        }

        # Norms
        layer_data["components"]["ln1"] = {"mean": float(l.ln1.weight.mean()), "std": float(l.ln1.weight.std())}
        layer_data["components"]["ln2"] = {"mean": float(l.ln2.weight.mean()), "std": float(l.ln2.weight.std())}

        genome["layers"].append(layer_data)
        if i % 2 == 0: print(f"  Layer {i} sequenced...")

    # 3. Unembed
    uw = model.unembed.weight.data
    genome["unembed"] = {
        "stats": get_stats(uw),
        "spectral": get_spectral_data(uw)
    }

    with open(output_file, "w") as f:
        json.dump(genome, f, indent=2)
    print(f"Sequencing complete: {output_file}")

if __name__ == "__main__":
    sequence_genome("model_finetuned_dpi_20m.pt", "GENOME_DPI.json")
    sequence_genome("model_xavier_20m.pt", "GENOME_XAVIER.json")
