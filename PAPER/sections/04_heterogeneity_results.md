### 4.2.3 Cross-Domain Generalization: Code Domain Evaluation

To evaluate the generalization capabilities of DPI, we tested the framework on the **Code-Heterogeneity** dataset (CodeSearchNet Python). 

**Lexical Structural Complexity**: Source code presents a unique challenge for initialization due to its rigid syntax, deep indentation hierarchies, and high-frequency keyword distribution. These features create an even more anisotropic latent space than natural language, making the initial weight configuration critical for convergence.

**Results: Convergence Acceleration Metrics**: On a 20.33M parameter model, DPI demonstrated substantial convergence speedups compared to the stochastic baseline. We performed a multi-seed evaluation ($N=3$) using a standardized learning rate ($LR=1 \times 10^{-4}$) across all domains to ensure comparability (Table 5).

**Table 5: Performance across Heterogeneous Data Domains (Source Code Challenge, N=3).**

| Metric (Step 500) | Xavier (Baseline) | DPI-14.1 (Exact SVD) | Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 8.294 $\pm$ 0.000 | **4.140 $\pm$ 0.023** | **-4.15** |
| **Perplexity** | 4000.7 $\pm$ 1.6 | **62.8 $\pm$ 1.5** | **63.7x better** |

*Note: Mean $\pm$ Standard Deviation for N=3 seeds. Perplexity is calculated as $exp(Loss)$. The **63.7x reduction in perplexity** highlights the massive structural advantage of DPI in the code domain, where stochastic methods struggle to bypass basic syntactic discovery within initial training steps.*

**Interpretation: Universal Statistical Priming Hypothesis**: The observation that DPI’s performance advantage is substantially larger on code than on natural language (where it is typically 3x to 5x) suggests that highly structured data benefits disproportionately from geometric pre-conditioning. The **Exact SVD** (Phase 0) successfully captured the grammar and indentation patterns of Python, allowing the model to bypass the "syntax-discovery" phase entirely. These findings indicate that DPI is a versatile pre-conditioning framework capable of adapting to diverse data topologies.
