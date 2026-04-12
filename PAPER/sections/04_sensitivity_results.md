### 4.3.3 Hyperparameter Robustness and Sensitivity

A common critique of deterministic initialization frameworks is the perceived reliance on specific architectural constants. To address this, we conducted an exhaustive grid search (triangulation) over 27 parameter combinations on a 20.33M model over one full epoch.

**Parameter Optimization Analysis**: The triangulation identified a broad plateau of high performance. The most efficient configuration was found with a Zipfian spectral warp ($\zeta$) of 1.0, a spectral gamma ($\gamma_0$) of 0.25, and a morph alpha ($\alpha$) of 0.45. This configuration reached a validation loss of 5.9466, representing the current lower bound for the DPI framework at this scale.

**Sensitivity Analysis**: Our analysis reveals that DPI is remarkably robust to hyperparameter variance. Performance remains stable across the $[1.0, 1.4]$ range for spectral warp, with a maximum loss variance of only 0.015. This suggests that the power-law alignment provided by the DCT basis is more significant than the specific warp factor. 

\vspace{1em}
![Figure 3: Warp Factor Sensitivity. Impact of the Zipfian spectral warp factor $\zeta$ on the final validation loss. The stability plateau indicates robustness to minor parameter tuning.](figures/sensitivity_zeta.png){width=80%}
\vspace{1em}

As illustrated in Figure 3, the loss surface remains notably flat across the tested warp range. Regarding spectral gamma, the model demonstrated a preference for light to moderate compression. Deviating toward excessively high ($>0.50$) or low ($<0.15$) gamma values causes a slight degradation in performance, confirming that the spectral bottleneck is a real physical constraint for most efficient manifold alignment.

\vspace{1em}
![Figure 4: Spectral Gamma Sensitivity. Validation loss as a function of the spectral gamma $\gamma_0$, demonstrating the optimal manifold compression range for efficient signal flow.](figures/sensitivity_gamma.png){width=80%}
\vspace{1em}

The sensitivity results for gamma (Figure 4) further reinforce the stability of the geometric prior. 

**Robustness Evaluation**: The "flatness" of the loss surface across these parameters indicates that DPI is a robust method. It provides a stable performance floor that is largely insensitive to minor tuning errors, making it a reliable and readily integrable protocol for industrial pre-training.
