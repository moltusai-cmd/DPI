### 4.2.2 Large-Scale Convergence Analysis (8.19-Billion Parameters)

To test the absolute limits of the DPI framework, we scaled our architecture to **8.19 Billion parameters** (40 layers, $d_{model}=4096$). Our objective was to measure "Gradient Conductivity" using a **Virtual Batch Size of 32** (via gradient accumulation), simulating professional pre-training conditions on a single consumer GPU (RTX 5080).

#### 4.2.2.1 Gradient Stagnation Analysis of Stochastic Baselines
We subjected an industry-standard **Xavier-Scaled** ($1/\sqrt{2L}$) baseline to a "Sudden Launch" protocol: 1,000 steps at $LR=10^{-4}$ with **0% warmup**. Even with a large virtual batch, the model remained paralyzed, with the Gradient Norm (GN) hovering at 0.14 and the validation loss stagnating at 9.69. These results suggest that stochastic noise at the 8B scale effectively acts as a signal insulator; without a significant warmup period to stabilize the initial weight manifold, the optimizer receives no actionable feedback for gradient updates.

#### 4.2.2.2 Spectral Alignment and the S-DPI Hybrid
In contrast, **DPI (PID-14)** exhibited immediate and robust gradient conductivity. We explored two distinct regimes of this initialization strategy. The first, **DPI Pur (Theoretical Purity)**, omitted depth-scaling and achieved a validation loss of 7.50 by Update 200. However, the high signal conductivity (GN > 6000) observed in this regime necessitates careful learning rate management to prevent eventual divergence. 

The second regime, **S-DPI (Industrial Hybrid)**, combines DPI with $1/\sqrt{2L}$ depth-scaling to prioritize stability. This configuration successfully stabilized the GN at approximately 470, allowing the model to reach a loss of 8.10 within the first 100 updates. These findings highlight the trade-offs between convergence speed and numerical stability at large scales.

#### 4.2.2.3 Quantitative Performance Metrics

Table 4: Gradient Conductivity and Stability at 8.19B scale (Batch Size 32).

| Configuration | Init Type | Learning Rate | GN (Avg) | Loss (U100) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Xavier-Scaled** | Stochastic | $10^{-4}$ | 0.14 | 9.69 | **Stagnated** |
| **DPI Pur** | Geometric | $10^{-5}$ | 6411.0 | **7.50*** | **Conductive** |
| **S-DPI Hybrid**| **DPI + $1/\sqrt{2L}$**| $10^{-4}$ | **478.3** | **8.10** | **Efficient** |
*\*DPI Pur data measured at Update 200 due to rapid convergence.*


#### 4.2.2.4 Robustness Analysis under Hardware Constraints
The 8.19B model was trained using 4-bit NormalFloat (NF4) quantization with CPU offloading of optimizer states. The sustained stability of the S-DPI hybrid under these conditions demonstrates that geometric initialization is resilient to the numerical noise introduced by extreme model compression. This resilience suggests that DPI is a viable candidate for professional-grade LLM training in hardware-constrained environments.
