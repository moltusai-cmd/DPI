# 4. Results and Discussion

In this section, we present a comprehensive empirical evaluation of the **Deterministic Pipeline Initialization (DPI)** framework across multiple scales, architectures, and data domains. 

Our experiments are designed to test three primary hypotheses:
1.  **Convergence Velocity**: That geometric pre-conditioning provides a significant and permanent speedup in information absorption compared to stochastic methods.
2.  **Zero-Warmup Stability**: That DPI-initialized manifolds possess the structural integrity to survive high-energy gradient updates from the first step of training.
3.  **Scale Invariance**: That the geometric constants identified at the 20M scale transfer successfully to billion-parameter architectures.

The following sub-sections detail our findings, organized into three thematic blocks: small-scale performance benchmarking, scaling and generalization analysis, and structural sensitivity investigations, as well as qualitative and quantitative analyses of the "Quantization Tax" and "Cross-Domain Generalization."
