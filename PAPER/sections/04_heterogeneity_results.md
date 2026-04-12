### 4.2.3 Cross-Domain Generalization: Code Domain Evaluation

To evaluate the generalization capabilities of DPI, we tested the framework on the **Code-Heterogeneity** dataset, consisting of high-density Python source code. 

**Lexical Structural Complexity**: Source code presents a unique challenge for initialization due to its rigid syntax, deep indentation hierarchies, and high-frequency keyword distribution. These features create an even more anisotropic latent space than natural language, making the initial weight configuration critical for convergence.

**Results: Convergence Acceleration Metrics**: On a 20.33M parameter model, DPI demonstrated substantial convergence speedups compared to the stochastic baseline. Specifically, the DPI model reached a level of syntactic understanding at Step 100 that the Xavier baseline failed to achieve even after 500 steps. These results suggest a greater than 5x acceleration in initial information absorption on heterogeneous data.

**Table 5: Performance across Heterogeneous Data Domains (Source Code Challenge).**

| Metric (Step 500) | Xavier (Baseline) | DPI (DPI-14) | Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 6.2374 | **3.6071** | **-2.63** |
| **Perplexity** | 511.5 | **36.8** | **13.9x better** |

*Note: Perplexity is calculated as $exp(Loss)$. The 13.9x reduction in perplexity highlights the massive structural advantage of DPI in the code domain.*

**Interpretation: Universal Statistical Priming Hypothesis**: The observation that DPI’s performance advantage is larger on code than on natural language suggests that highly structured data benefits disproportionately from geometric pre-conditioning. The singular value decomposition (SVD) applied during Phase 0 successfully captured the grammar of Python from a limited sample, allowing the model to bypass the syntax-discovery phase. These findings indicate that DPI is a versatile pre-conditioning framework capable of adapting to diverse data topologies.
