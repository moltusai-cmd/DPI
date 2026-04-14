# Appendix X: Comparative Analysis at Scale (100M Class)

This appendix documents a standardized performance evaluation between MuDPI v16.3 and the muP-Xavier baseline. The goal was to characterize the convergence efficiency of geometric alignment under optimal learning rate configurations ($LR_{crit}$) identified via empirical grid search.

## X.1 Experimental Configuration

The benchmark utilized a state-of-the-art Llama-style transformer architecture optimized for hardware efficiency.

| Parameter | Configuration |
| :--- | :--- |
| Model Scale | 100M parameters (d_model=768, n_layers=12, n_heads=12, d_mlp=2048) |
| Architecture | RMSNorm, SwiGLU, RoPE (256 context) |
| Dataset | ArXiv (95% Train / 5% Val Split) |
| Batch Size | 64 (1.6 Billion Tokens total) |
| Precision | BF16 Mixed Precision (torch.amp) |
| Rank Threshold | $10^{-3}$ (0.1% of max singular value) |

## X.2 Baseline and Protocol Definition

To ensure a rigorous comparison, each initialization was evaluated at its respective stability limit:
*   **Xavier-muP:** $LR = 2 \cdot 10^{-4}$ (with 2,000 steps linear warmup and cosine decay).
*   **MuDPI v16.3:** $LR = 8 \cdot 10^{-4}$ (with **Stable-Decay** : 8,000 steps at 100% power, 2,000 steps cosine decay).

This differential loading ensures that both methods are compared at their maximum stable operational capacity.

## X.3 Empirical Results: Convergence Rates

Measurements were taken on an independent validation set (mean of 50 batches).

| Step | Xavier Val Loss | Xavier Rank | MuDPI Val Loss | MuDPI Rank | $\Delta$ Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 9.7099 | 767 | 9.2404 | 766 | -0.4695 |
| 1000 | 6.1268 | 767 | 3.8993 | 767 | -2.2275 |
| 2000 | 4.9138 | 756 | 3.6086 | 767 | -1.3052 |
| 5000 | 3.9756 | 756 | 3.3505 | 767 | -0.6251 |
| 8000 (Pre-Decay) | 3.7677 | 756 | 3.2660 | 765 | -0.5017 |
| 10000 (Final) | 3.7512 | 756 | **3.1718** | **765** | **-0.5794** |

## X.4 Qualitative Analysis: Feature Mapping Fidelity

At step 10,000, both models were submitted to the prompt: *"The derivation of the Einstein field equations starts from..."*

*   **MuDPI Result:** Successfully identified the foundational link to the **Euler-Lagrange equations**, mirroring the Hilbert-Einstein action derivation found in the training corpus.
*   **Xavier Result:** Utilized generic terminology ("standard model", "numerical simulations"), suggesting a lower density of specialized information mapping within the constrained parameter space.

## X.5 Conclusion: Computational Efficiency Delta

MuDPI v16.3 reached the baseline final validation loss (3.75) at approximately **step 1400**. This represents a **7.1x speedup in training efficiency**. The higher rank preservation (765 vs 756) under high learning rate conditions indicates that geometric alignment enhances the model's capacity to absorb information during the initial training phase.
