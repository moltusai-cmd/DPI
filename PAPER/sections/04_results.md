# 4. Results and Discussion

Our experiments are organized into three thematic blocks: performance benchmarking at small scale, scaling and generalization analysis, and structural sensitivity investigations. These experiments are designed to test three primary hypotheses:
1.  **Convergence Velocity**: That geometric pre-conditioning provides a significant and permanent speedup in information absorption compared to stochastic methods.
2.  **Zero-Warmup Stability**: That DPI-initialized manifolds possess the structural integrity to survive high-energy gradient updates from the first step of training.
3.  **Scale Invariance**: That the geometric constants identified at the 20M scale transfer successfully to billion-parameter architectures.

The following sub-sections detail our findings.
