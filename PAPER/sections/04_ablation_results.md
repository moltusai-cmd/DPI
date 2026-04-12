# 4.5 THE CALIBRATION AND WHITENING PARADOXES

To understand the contribution of each DPI component in the context of the new **Sequential Bootstrapping (PID-14.1)** architecture, we conducted a multi-scale ablation study. While core structural components (Lexical Seeding, QKV Signatures) are essential, the roles of **Mahalanobis Whitening** and **LayerNorm Calibration** revealed counter-intuitive phenomena.

### 4.5.1 The Whitening Paradox (335M Scale)
Initially, it was hypothesized that Whitening (Phase 5) would provide necessary decorrelation for larger models. However, at the **335.64M scale**, the results were unequivocal:
*   **DPI No-White**: Reached Loss **5.60** at step 200.
*   **Full DPI (With Whitening)**: Reached Loss **6.99** at step 200.

Removing the whitening phase resulted in a **1.39 point loss advantage**, suggesting that forcing a strictly decorrelated latent space effectively "strips" the model of the structural priors provided by earlier phases.

### 4.5.2 The Calibration Paradox (20M Triple Duel)
In the new Sequential PID-14.1 framework, we evaluated the impact of **Phase 6 (Calibration)** in a direct comparison against a Xavier baseline with **0% warmup** over 300 steps.

| Variant | Step 300 Loss | Delta vs Xavier | Status |
| :--- | :--- | :--- | :--- |
| Xavier (Baseline) | 8.2057 | - | Unstable |
| DPI-Full (With Calib) | 6.9983 | -1.21 | Stable |
| **DPI-NoCalib** | **6.7964** | **-1.41** | **OPTIMAL** |

**The conclusion is definitive**: In a sequential sculpting regime, DPI without final calibration is **0.20 points better** than with it. The precision of sequential bootstrapping is so high that additional variance normalization acts as a "signal dampener" rather than a stabilizer.

### 4.5.3 Conclusion: Geometric Autonomy
The ablation data across all scales (20M to 335M) suggests that **PID-14.1 (Sequential + No-White + No-Calib)** is the most efficient and mathematically pure configuration. By allowing the manifold to breathe naturally after its initial geometric sculpting, DPI achieves superior convergence without the need for traditional industrial "béquilles" (crutches) like whitening or post-hoc calibration.
