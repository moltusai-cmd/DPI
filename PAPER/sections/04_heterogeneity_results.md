### 4.2.3 Cross-Domain Generalization: Code Domain Evaluation

To test whether DPI’s lexical seeding and spectral priors are over-fitted to the homogeneous distributions of natural language, we evaluated the framework on a dataset with radically different statistical properties: **Python Source Code (CodeSearchNet)**.

#### 4.2.3.1 Structural Complexity in Source Code
Source code presents a unique challenge for initialization due to its rigid syntax, deep indentation hierarchies, and high-frequency keyword distribution. These features create an even more anisotropic latent space than natural language, making the initial weight configuration critical for convergence.

#### 4.2.3.2 Results: Convergence Acceleration Metrics
On a 20.33M parameter model, DPI demonstrated substantial convergence speedups compared to the stochastic baseline. Specifically, the DPI model reached a level of syntactic understanding at Step 100 that the Xavier baseline failed to achieve even after 500 steps. These results suggest a greater than 5x acceleration in initial information absorption on heterogeneous data.

Performance across Heterogeneous Data Domains (Source Code Challenge).

| Metric (Step 500) | Xavier (Baseline) | DPI (PID-14) | Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 6.2374 | **3.6071** | **-2.63** |
| **Perplexity** | 511.5 | **36.8** | **13.9x better** |

#### 4.2.3.3 Interpretation: Universal Statistical Priming Hypothesis
The fact that DPI’s performance advantage is larger on code than on natural language suggests that highly structured data benefits disproportionately from geometric pre-conditioning. The singular value decomposition (SVD) applied during Phase 0 successfully captured the grammar of Python from a limited sample, allowing the model to bypass the syntax-discovery phase. This findings indicate that DPI is a universal pre-conditioning framework capable of adapting to diverse data topologies.
## 4.3 Structural and Sensitivity Investigations

### 4.3.1 Robustness to Data Sampling Density (Phase 0)

A potential criticism of DPI’s lexical seeding phase is the perceived logistical overhead of constructing co-occurrence matrices for large-scale corpora. To address this, we conducted a sensitivity analysis on sampling density over a sustained training interval of 300 steps.

#### 4.3.1.1 The Sparse Initialization Experiment
We compared two initialization regimes for a 20M parameter model to determine the minimum data requirements for Phase 0. The first regime, **Ultra-Sparse**, computed the lexical seeding on only 100 lines of raw text, while the second regime, **Standard**, used 10,000 lines.

#### 4.3.1.2 Results: Invariant Semantic Priors
The convergence trajectories were nearly identical throughout the training process, with a negligible loss delta of 0.053 at Step 300 (Ultra-Sparse: 7.06 vs. Standard: 7.01). These results prove that the macroscopic geometric structure of language is captured almost instantly, suggesting that DPI does not require processing the full training corpus for initialization.

#### 4.3.1.3 Scalability Assessment
The finding that a vanishingly small sample is sufficient to provide the structural priors required for immediate gradient conductivity has significant implications for large-scale pre-training. It effectively reduces the lexical seeding overhead to near-zero, confirming DPI’s logistical viability for industrial-scale applications.
