# Appendix: Technical Nomenclature

To ensure clarity across all experimental scales and architectural variants, we formally define the core abbreviations and framework components employed in this work (Table A1).

**Table A1: Glossary of Technical Abbreviations and Framework Components.**

| Abbreviation | Full Term | Functional Definition |
| :--- | :--- | :--- |
| **DPI** | Deterministic Pipeline Initialization | The overarching framework replacing stochastic noise with geometric priors. |
| **DPI-14.1** | Sequential Bootstrapping | The specific architecture version where layer $l$ is initialized using activations from layer $l-1$. |
| **S-DPI** | Scaled-DPI Hybrid | A configuration combining DPI geometric priors with $1/\sqrt{2L}$ depth-scaling for large-scale stability. |
| **GN** | Gradient Norm | A metric of signal conductivity; high GN indicates effective backpropagation of the loss signal. |
| **$\mu$P** | Maximal Update Parametrization | A mathematical framework for width-independent hyperparameter scaling. |
| **SVD** | Singular Value Decomposition | The linear algebra operation used in Phase 0 to extract lexical structure from sparse data. |
| **DCT** | Discrete Cosine Transform | The frequency-domain basis used for spectral warping in Phase 2. |
| **NF4** | 4-bit NormalFloat | An information-theoretically optimal quantization format used for 8B-scale evaluation. |

*Note: These definitions are applied consistently across all experimental scales (20M to 8.19B parameters).*
