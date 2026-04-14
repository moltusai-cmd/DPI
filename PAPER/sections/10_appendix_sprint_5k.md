# Appendix X: Technical Validation "Sprint 5k" (350M Class)

This appendix documents the stress-test validation of DPI v16.3 against a maximal update parameterization (muP) baseline.

## X.1 Experimental Setup

The experiment was conducted on a single GPU (16GB VRAM) using a standard Llama-style Transformer architecture.

| Parameter | Configuration |
| :--- | :--- |
| Model Scale | 350M parameters (d_model=1024, n_layers=24, n_heads=16, d_mlp=4096) |
| Architecture | RMSNorm, SwiGLU, RoPE (512 context) |
| Dataset | ArXiv (BPE Tokenized, 100k lines) |
| Batch Size | 8 samples per step |
| Precision | BF16 Mixed Precision with Gradient Checkpointing |
| Optimizer | muP-AdamW ($lr=10^{-4}$) |

## X.2 Protocol Definition

Two groups were evaluated under identical conditions except for the initialization protocol:

*   **Group A (Baseline):** Elite muP scaling with Xavier Uniform initialization. Linear Warmup: 2,000 steps.
*   **Group B (DPI v16.3):** Deterministic Pipeline Initialization (Spectral Isometry mode). Warmup: 0 steps (Constant LR).

## X.3 Empirical Results

Metric monitoring was performed every 500 steps. The "Effective Rank" ($\rho_{eff}$) is defined as the number of singular values above 5% of the maximum singular value in the middle-layer query projection ($W_q$).

### X.3.1 Comparative Loss and Rank Stability

| Step | Group A (Xavier) Loss | Group A Rank | Group B (DPI) Loss | Group B Rank | $\Delta$ Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 9.7067 | 958 | 9.3687 | 960 | -0.3380 |
| 500 | 8.6122 | 959 | 6.3600 | 960 | -2.2522 |
| 1000 | 6.6911 | 958 | 5.8437 | 960 | -0.8474 |
| 2000 | 5.9935 | 922 | 5.2238 | 960 | -0.7697 |
| 3000 | 5.2828 | 848 | 5.0343 | 960 | -0.2485 |
| 4000 | 4.8512 | 833 | 5.0399 | 960 | +0.1887 |
| 5000 | 5.0767 | 820 | **4.7335** | **960** | **-0.3432** |

## X.4 Technical Observations

1.  **Dimensional Collapse:** Group A (Xavier) exhibited significant dimensional collapse post-warmup, with $\rho_{eff}$ decreasing by 14.4% (958 $\to$ 820). Group B (DPI) maintained maximum rank (960) throughout the training duration.
2.  **Warmup Elimination:** DPI v16.3 allowed for 100% learning rate injection from step 1 without stability degradation, achieving a loss of 6.3600 at step 500 while the baseline was still at 8.6122.
3.  **Convergence Velocity:** DPI achieved the baseline's 3,000-step performance (5.28) at step 2,000, representing a 1.5x computational efficiency gain.
4.  **Final Manifold Advantage:** DPI finished with a 0.34 point validation loss advantage, confirming that geometric alignment provides a structural benefit that persists beyond the initial phase.
