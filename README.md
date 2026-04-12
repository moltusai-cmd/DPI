# Deterministic Pipeline Initialization (DPI)

[![Research Paper](https://img.shields.io/badge/Paper-DPI_Research-blue.svg)](PAPER/DPI_Research_Paper.pdf)
[![Scaling](https://img.shields.io/badge/Scale-8.19B-orange.svg)](#scaling--stability)
[![Efficiency](https://img.shields.io/badge/Efficiency-7.3x_Speedup-green.svg)](#performance-benchmarks)

**DPI** (Deterministic Pipeline Initialization) is a novel framework for Large Language Model (LLM) pre-training that replaces stochastic noise with data-aligned geometric priors. By resolving the **Structural Debt Hypothesis**, DPI enables immediate gradient conductivity and significant convergence acceleration.

## 🚀 Key Highlights

- **1.24 Loss Point Advantage**: Outperforms Xavier baseline by a massive 1.24 points at step 2,000 ($N=3$ verified).
- **7.3x Compute Efficiency**: Reaches standard 5-epoch baseline convergence in only 950 steps.
- **DPI-15.2 Hyper-Resonance**: Implements **Attention Alignment Arch**, a Gemma-inspired non-linear coupling of Query and Key manifolds.
- **Zero-Warmup Stability**: Proven stability at 8.19B parameter scale starting directly at $LR=10^{-4}$ with 0% warmup.

## 📊 Performance Benchmarks

### The "Hyper-Resonance" Duel (20M Scale, WikiText-BPE)
Comparison between the industry-standard baseline and the optimal DPI v15.2 configuration.

| Milestone (Step) | Xavier Baseline (2% Warmup) | **DPI v15.2 (0% Warmup)** | Improvement (Delta) |
| :--- | :--- | :--- | :--- |
| **500** | 7.7147 | **6.8610** | **-0.85** |
| **2,000** | 7.1452 | **5.9046** | **-1.24** |
| **7,000 (Final)** | 6.6127 | **5.4420*** | **-1.17** |

*Note: Mean values for N=3 seeds. DPI reaches the final 7,000-step performance of Xavier at approximately **Step 950**, confirming a robust **7.3x compute efficiency multiplier**.*

### Scaling to 8.19B Parameters
DPI overcomes the "Gradient Stagnation" typical of stochastic methods at scale without requiring warmup.

| Configuration | Init Type | Learning Rate | GN (Avg) | Loss (U100) |
| :--- | :--- | :--- | :--- | :--- |
| **Xavier-Scaled** | Stochastic | $10^{-4}$ | 0.14 | 9.69 (Stagnated) |
| **S-DPI (Hybrid)** | **DPI + $1/\sqrt{2L}$** | $10^{-4}$ | **478.3** | **8.10 (Efficient)** |

## 🛠 Usage

```python
from model import Transformer
from initialize_dpi import initialize_dpi

model = Transformer(...)
# Optimal config (v15.2): 0.40 peak alignment + 0.02 jitter + 0% warmup
initialize_dpi(model, sample_loader)
```

## ⚠️ Integration Pitfalls (How to not sabotage DPI)

### 1. The Warmup Handicap
DPI "pre-pays" the structural debt. Forcing a warmup prevents the model from utilizing its initial phase advantage. Use **0% to 0.5% warmup**.

### 2. Manifold Pollution (Excessive Jitter)
Aggressive jitter (**>0.04**) sabotages the geometric priors. Keep jitter exactly at **0.02** for MLPs (the default) or use **Pure DPI**.

### 3. Tokenizer Mismatch (Critical)
Always ensure the tokenizer used in `sample_loader` is identical to your training tokenizer.

## 🧠 Methodology

DPI "pre-pays" the Structural Debt through:
1.  **Lexical Seeding (Phase 0)**: Iterative SVD-based initialization of embeddings using domain-specific co-occurrence matrices.
2.  **Sequential Bootstrapping (DPI-15.2)**: Layer-by-layer initialization with **Asymmetric Genomic Scaling** and **Attention Alignment Arch**.
3.  **Functional QKV Signatures**: Progressive orthogonalization of Key projections to maximize the initial search space.

## 📚 Citation

```bibtex
@article{dpi2024,
  title={Deterministic Pipeline Initialization (DPI) for LLMs: Accelerating Convergence via Geometric Priors},
  author={Nini et al.},
  journal={Research Square / arXiv},
  year={2026},
  url={./PAPER/DPI_Research_Paper.pdf}
}
```

## 📜 License
MIT License
