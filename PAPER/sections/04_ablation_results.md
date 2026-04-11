# 4.5 ABLATION STUDY AND THE WHITENING PARADOX

To understand the contribution of each DPI component, we conducted a multi-scale ablation study. While most components (Lexical Seeding, Calibration) provided consistent gains, the role of **Mahalanobis Whitening (Phase 5)** revealed a counter-intuitive phenomenon.

### 4.5.1 Component Importance (20M Scale)
At the 20M parameter scale, the removal of core structural components led to significant performance degradation over one epoch.

| Variant | Final Loss | Delta vs Full | Interpretation |
| :--- | :--- | :--- | :--- |
| **No Whitening** | **6.1424** | **-0.09** | **Superior Performance** |
| Full PID-14 | 6.2350 | - | Baseline |
| No Phase 0 | 6.6875 | +0.45 | Semantic Lag |
| No Calibration | 7.0452 | +0.81 | Instability |

### 4.5.2 The 335M Ablation: Empirical Validation
Initially, it was hypothesized that Mahalanobis Whitening would act as a "Scaling Guardian," providing necessary decorrelation for larger models. We tested this by comparing Full DPI against a No-Whitening variant on a **335.64M parameter** model (200 steps).

*   **Full DPI (With Whitening)**: Reached Loss **6.99** at step 200.
*   **DPI No-White**: Reached Loss **5.60** at step 200.

**The result is unequivocal**: Removing the whitening phase resulted in a **1.39 point loss advantage** at the 335M scale.

### 4.5.3 Conclusion: Preserving Semantic Correlation
The ablation data suggests that the internal activations of a Transformer benefit from maintaining local semantic correlations. Forcing a strictly decorrelated latent space via whitening—while theoretically sound for variance control—effectively "strips" the model of the structural priors provided by the earlier DPI phases.

We conclude that **Phase 0 (Embeddings) and Phase 6 (Calibration)** are the essential pillars of DPI, while Whitening is counter-productive across all tested scales. Consequently, the production version of DPI (PID-14) defaults to a "No-White" configuration.
