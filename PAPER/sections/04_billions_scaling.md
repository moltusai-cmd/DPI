### 4.2.2 Large-Scale Convergence Analysis (8.19-Billion Parameters)

To evaluate the scaling limits of the DPI framework, we implemented the architecture at **8.19 Billion parameters** (40 layers, $d_{model}=4096$). The evaluation focused on measuring "Gradient Conductivity" using a **Virtual Batch Size of 32** (via gradient accumulation), simulating production-scale training conditions on consumer-grade hardware (RTX 5080).

**Gradient Stagnation Analysis of Stochastic Baselines**: We subjected an industry-standard **Xavier-Scaled** ($1/\sqrt{2L}$) baseline to a **Direct Optimization Protocol**: 1,000 steps at $LR=10^{-4}$ with **0% warmup**. Even with a large virtual batch, the model exhibited stagnation, with the Gradient Norm (GN) hovering at 0.14 and the validation loss failing to diverge from 9.69. These results indicate that stochastic noise at the 8B scale acts as a signal insulator; without a sustained warmup period to stabilize the initial weight manifold, the optimizer lacks sufficient signal for meaningful parameter updates.

**Spectral Alignment and the S-DPI Hybrid**: In contrast, **DPI (DPI-14)** demonstrated immediate and robust gradient conductivity. We analyzed two distinct initialization regimes. The first, **Unconstrained Geometric Initialization (UGI)**, omitted depth-scaling and reached a validation loss of 7.50 by Update 200. However, the high signal conductivity (GN > 6000) observed in this regime requires careful learning rate management to prevent eventual numerical divergence. 

The second regime, **S-DPI (Scaled DPI)**, represents our production-ready configuration. By integrating DPI with $1/\sqrt{2L}$ depth-scaling, we successfully harmonized high-velocity geometric alignment with numerical stability. This configuration stabilized the Gradient Norm at **~478**, enabling the model to reach a validation loss of **8.10** within 100 updates, effectively bypassing the stagnation observed in stochastic models.

**Table 4: Stability and Convergence at the 8.19B Scale (Batch Size 32).**

| Configuration | Init Strategy | GN (Mean) | Loss (U100) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Xavier-Scaled** | Stochastic | 0.14 | 9.69 | **Stagnated** |
| **UGI (Unconstrained)** | Geometric | > 6000 | **7.50*** | **High Variance** |
| **S-DPI (Hybrid)** | **Geometric + Scaling** | **478.3** | **8.10** | **Stable/Optimal** |

*Note: The UGI configuration, while demonstrating extreme signal conductivity, exhibited symptoms of numerical instability at higher learning rates, necessitating the depth-scaling modulation employed in S-DPI.*

**Robustness Analysis under Hardware Constraints**: The 8.19B model was evaluated using 4-bit NormalFloat (NF4) quantization with CPU offloading of optimizer states. The sustained stability of the S-DPI hybrid under these conditions demonstrates that geometric initialization is resilient to the numerical noise introduced by extreme model compression. This resilience establishes DPI as a viable protocol for large-scale model training in hardware-constrained environments.
