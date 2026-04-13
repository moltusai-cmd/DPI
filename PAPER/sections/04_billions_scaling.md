### 4.2.2 Large-Scale Convergence Analysis (8.19-Billion Parameters)

To test the scaling limits of the DPI framework, we scaled our architecture to **8.19 Billion parameters** (40 layers, $d_{model}=4096$). Our objective was to measure "Gradient Conductivity" using a **Virtual Batch Size of 32** (via gradient accumulation), simulating professional pre-training conditions on a single consumer GPU (RTX 5080).

**Gradient Stagnation Analysis of Stochastic Baselines**: We subjected an industry-standard **Xavier-Scaled** ($1/\sqrt{2L}$) baseline to a "Sudden Launch" protocol: 1,000 steps at $LR=10^{-4}$ with **0% warmup**. Even with a large virtual batch, the model remained paralyzed, with the Gradient Norm (GN) hovering at 0.14 and the validation loss stagnating at 9.69. These results suggest that stochastic noise at the 8B scale effectively acts as a signal insulator; without a significant warmup period to stabilize the initial weight manifold, the optimizer receives no actionable feedback for gradient updates.

**Spectral Alignment and the S-DPI Hybrid**: In contrast, **DPI (DPI-14)** exhibited immediate and robust gradient conductivity. We explored two distinct regimes of this initialization strategy. The first, **DPI Pur (Theoretical Purity)**, omitted depth-scaling and achieved a validation loss of 7.50 by Update 200. However, the high signal conductivity (GN > 6000) observed in this regime necessitates careful learning rate management to prevent eventual divergence. 

The second regime, **S-DPI (Scaled DPI)**, represents our production-ready configuration. By integrating DPI with $1/\sqrt{2L}$ depth-scaling, we successfully harmonized high-speed geometric alignment with numerical stability. This configuration stabilized the Gradient Norm at a robust **~478**, enabling the model to reach a validation loss of **8.10** within only 100 updates, effectively bypassing the stagnation observed in stochastic models.

**Table 4: Stability and Convergence at the 8.19B Scale (Batch Size 32).**

| Configuration | Init Strategy | GN (Mean) | Loss (U100) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Xavier-Scaled** | Stochastic | 0.14 | 9.69 | **Stagnated** |
| **DPI Pure** | Geometric | > 6000 | **7.50*** | **Unstable** |
| **S-DPI (Hybrid)** | **Geometric + Scaling** | **478.3** | **8.10** | **Stable/Optimal** |

*Note: The DPI Pure configuration, while demonstrating extreme signal conductivity, showed signs of numerical divergence at higher learning rates, necessitating the depth-scaling modulation used in S-DPI.*

**Robustness Analysis under Hardware Constraints**: The 8.19B model was trained using 4-bit NormalFloat (NF4) quantization with CPU offloading of optimizer states. The sustained stability of the S-DPI hybrid under these conditions demonstrates that geometric initialization is resilient to the numerical noise introduced by extreme model compression. This resilience suggests that DPI is a viable candidate for professional-grade LLM training in hardware-constrained environments.
