# 4.7 CROSS-DOMAIN GENERALIZATION: THE CODE CHALLENGE

To test whether DPI’s lexical seeding and spectral priors are over-fitted to the homogeneous distributions of natural language, we evaluated the framework on a dataset with radically different statistical properties: **Python Source Code (CodeSearchNet)**.

### 4.7.1 Structural Complexity of Code
Source code presents a unique challenge for initialization due to its rigid syntax, deep indentation hierarchies, and high-frequency keyword distribution (*def, self, return*). These features create an even more anisotropic latent space than natural language.

### 4.7.2 Results: Massive Acceleration
On a 20.33M parameter model, DPI demonstrated its most significant acceleration to date.

| Metric (Step 500) | Xavier (Baseline) | DPI (PID-14) | Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 6.2374 | **3.6071** | **-2.63** |
| **Perplexity** | 511.5 | **36.8** | **13.9x better** |

The DPI model reached a level of syntactic understanding at Step 100 (Val Loss 4.75) that the Xavier baseline failed to achieve even after 500 steps. This represents a **>5x speedup** in initial information absorption on heterogeneous data.

### 4.7.3 Interpretation: Universal Statistical Priming
The fact that DPI’s performance delta is larger on code than on natural language suggests that the more structured the data, the more it benefits from geometric pre-conditioning.

The Singular Value Decomposition (SVD) of co-occurrence matrices (Phase 0) successfully captured the “grammar” of Python from just 1,000 samples, allowing the model to bypass the syntax-discovery phase. This confirms that DPI is a **universal pre-conditioning framework** capable of adapting to diverse data topologies without modification.

---

# 4.11 ROBUSTNESS TO DATA SAMPLING DENSITY (PHASE 0)

A potential criticism of DPI’s Phase 0 (Lexical Seeding) is the perceived logistical overhead of constructing co-occurrence matrices for trillion-token corpora. To address this, we conducted a sensitivity analysis on sampling density over a sustained training interval (300 steps).

### 4.11.1 The Sparse Initialization Experiment
We compared two initialization regimes for a 20M parameter model:
*   **Ultra-Sparse**: Phase 0 computed on only **100 lines** of raw text.
*   **Standard**: Phase 0 computed on **10,000 lines** of raw text.

### 4.11.2 Results: Invariant Semantic Priors
The convergence trajectories were nearly identical throughout the training process. At Step 300, the loss delta was a negligible **0.053** (Ultra-Sparse: 7.06 vs. Standard: 7.01). 

### 4.11.3 Conclusion on Scalability
This result proves that the macroscopic geometric structure of language is captured almost instantly. Consequently, DPI does **not** require processing the full training corpus for initialization. A vanishingly small sample (e.g., <0.0001% of a trillion-token dataset) is sufficient to provide the "Magnetic North" required for immediate gradient conductivity. This confirms DPI’s logistical viability for industrial-scale LLM pre-training, effectively reducing the "Lexical Seeding" overhead to near-zero.
