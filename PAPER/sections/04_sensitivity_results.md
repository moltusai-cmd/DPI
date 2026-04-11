# 4.6 HYPERPARAMETER ROBUSTNESS AND SENSITIVITY

A common critique of deterministic initialization frameworks is the perceived reliance on "magic constants." To address this, we conducted an exhaustive grid search (triangulation) over 27 parameter combinations on a 20.33M model over one full epoch.

### 4.6.1 Optimal Sweet Spot
The triangulation identified a broad plateau of high performance. The optimal configuration was found at:
*   **Zipfian Warp ($\zeta$)**: 1.0 (Standard DCT)
*   **Spectral Gamma ($\gamma_0$)**: 0.25 (Light Spectral Compression)
*   **Morph Alpha ($\alpha$)**: 0.45 (Pronounced Semantic Transition)

This configuration reached a validation loss of **5.9466**, representing the current lower bound for the DPI framework at this scale.

### 4.6.2 Sensitivity Analysis
Our analysis reveals that DPI is remarkably robust to hyperparameter variance.

![Sensitivity to Zipf Warp](figures/sensitivity_zeta.png)

*   **Spectral Warp ($\zeta$)**: Performance remains stable across the $[1.0, 1.4]$ range, with a maximum loss variance of only **0.015**. This suggests that the power-law alignment provided by the DCT basis is more important than the specific warp factor.

![Sensitivity to Spectral Gamma](figures/sensitivity_gamma.png)

*   **Spectral Gamma ($\gamma_0$)**: The model prefers light to moderate compression. S'écarter vers des valeurs de gamma trop élevées ($>0.50$) or trop basses ($<0.15$) causes a slight degradation, confirming that the **Spectral Bottleneck** is a real physical constraint for optimal manifold alignment.

### 4.6.3 Conclusion on Robustness
The "flatness" of the loss surface across these parameters proves that DPI is not a "brittle" method. It provides a stable performance floor that is largely insensitive to minor tuning errors, making it a reliable plug-and-play solution for industrial pre-training.
