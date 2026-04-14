# 4. Results and Discussion

Our experiments are organized into three thematic blocks: performance benchmarking at the 350M scale against elite $\mu$P baselines, long-term scaling analysis, and structural sensitivity investigations. These experiments are designed to test three primary hypotheses:

1.  **Geometric Dominance**: That DPI-initialized models achieve lower validation loss than standard $\mu$P-Xavier baselines, even when the latter are tuned using industry-standard linear warmup.
2.  **Zero-Warmup Stability**: That DPI-initialized manifolds possess the structural integrity to survive high-energy gradient updates ($LR=10^{-4}$) from the first step of training.
3.  **Dimensional Integrity**: That DPI prevents "Dimensional Collapse" in late-stage layers, maintaining a higher effective rank throughout the training duration compared to stochastic models.

The cornerstone of our validation is the **Sprint 5k (350M Class)** benchmark (see Appendix X), which demonstrates a 1.5x efficiency multiplier over the official $\mu$P "Elite Scaling" baseline while maintaining a significantly deeper loss trajectory.
