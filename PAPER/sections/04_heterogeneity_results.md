### 4.2.3 Cross-Domain Generalization: Code Domain Evaluation

To test whether DPI’s lexical seeding and spectral priors are over-fitted to the homogeneous distributions of natural language, we evaluated the framework on a dataset with radically different statistical properties: **Python Source Code (CodeSearchNet)**.

#### 4.2.3.1 Structural Complexity in Source Code
Source code presents a unique challenge for initialization due to its rigid syntax, deep indentation hierarchies, and high-frequency keyword distribution. These features create an even more anisotropic latent space than natural language, making the initial weight configuration critical for convergence.

#### 4.2.3.2 Results: Convergence Acceleration Metrics
On a 20.33M parameter model, DPI demonstrated substantial convergence speedups compared to the stochastic baseline. Specifically, the DPI model reached a level of syntactic understanding at Step 100 that the Xavier baseline failed to achieve even after 500 steps. These results suggest a greater than 5x acceleration in initial information absorption on heterogeneous data.

**Table 5: Performance across Heterogeneous Data Domains (Source Code Challenge).**

| Metric (Step 500) | Xavier (Baseline) | DPI (PID-14) | Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 6.2374 | **3.6071** | **-2.63** |
| **Perplexity** | 511.5 | **36.8** | **13.9x better** |

*Note: Perplexity is calculated as $exp(Loss)$. The 13.9x reduction in perplexity highlights the massive structural advantage of DPI in the code domain.*

#### 4.3.2.3 Interpretation: Universal Statistical Priming Hypothesis
The fact that DPI’s performance advantage is larger on code than on natural language suggests that highly structured data benefits disproportionately from geometric pre-conditioning. The singular value decomposition (SVD) applied during Phase 0 successfully captured the grammar of Python from a limited sample, allowing the model to bypass the syntax-discovery phase. This findings indicate that DPI is a universal pre-conditioning framework capable of adapting to diverse data topologies.
