### 4.3.2 Component Ablation: Spectral Whitening and Calibration Convergence

To understand the contribution of each DPI component in the context of the new **Sequential Bootstrapping (DPI-14.1)** architecture, we conducted a multi-scale ablation study. While core structural components, such as lexical seeding and differentiated QKV signatures, are essential, the roles of Mahalanobis whitening and LayerNorm calibration revealed counter-intuitive phenomena.

**Impact of Phase 5 Spectral Whitening (335M Scale)**: Initially, it was hypothesized that Phase 5 whitening would provide necessary decorrelation for larger models. However, at the 335.64M scale, the results were notable: the DPI model without whitening reached a validation loss of 5.60 at Step 200, whereas the full DPI model with whitening achieved a loss of 6.99. 

This 1.39 point loss advantage suggests that forcing a strictly decorrelated latent space may effectively "strip" the model of the structural priors provided by earlier initialization phases. Consequently, allowing the manifold to maintain its natural spectral density appears to be more beneficial for convergence at intermediate scales.

**Impact of Phase 6 Layer Calibration (20M Triple Duel)**: In the new sequential DPI-14.1 framework, we evaluated the impact of Phase 6 calibration in a direct comparison against a Xavier baseline with 0% warmup over 300 steps. The results indicate that while the DPI-Full model with calibration achieved a stable loss of 6.99, the DPI-NoCalib variant achieved a superior loss of 6.79. 

Results indicate a clear performance advantage for the non-calibrated variant, suggesting that the precision of sequential bootstrapping is sufficiently high that additional variance normalization acts as a signal dampener rather than a stabilizer.

**Table 6: Impact of Phase 6 Calibration on Sequential Initialization stability.**

| Variant | Step 300 Loss | Delta vs Xavier | Status |
| :--- | :--- | :--- | :--- |
| Xavier (Baseline) | 8.2057 | - | Unstable |
| DPI-Full (With Calib) | 6.9983 | -1.21 | Stable |
| **DPI-NoCalib** | **6.7964** | **-1.41** | **Efficient** |

*Note: All tests performed at 20.33M scale with 0% warmup. The No-Calib variant provides the strongest signal flow.*

**Synthesis of Geometric Autonomy in DPI-14.1**: The ablation data across all scales (20M to 335M) suggests that the DPI-14.1 (Sequential + No-White + No-Calib) configuration is efficient and mathematically consistent. By allowing the manifold to evolve naturally after its initial geometric pre-conditioning, DPI achieves superior convergence without the need for traditional empirical stabilization techniques like whitening or post-hoc calibration.
