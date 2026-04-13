### 4.3.2 Component Ablation: Spectral Density vs. Forced Decorrelation

To isolate the individual contributions of the DPI components, we conducted a systematic ablation study. The results reveal a critical theoretical distinction between **statistical decorrelation** (whitening) and **geometric prior preservation**.

**Ablation of Spectral Whitening (335M Scale)**:
We evaluated the impact of Phase 5 Mahalanobis whitening—a technique traditionally used to ensure feature decorrelation. At the 335.64M parameter scale, we observed that the **Unwhitened DPI** configuration achieved a validation loss of **5.60** (Step 200), significantly outperforming the **Full-Whitened** variant (Loss 6.99).

This **1.39 point performance delta** indicates that the structural priors established in Phases 0-2 (lexical seeding and spectral warping) are more critical for convergence than isotropic decorrelation. Forced whitening effectively acts as a "structural erasure" mechanism, stripping the manifold of the natural, anisotropic clusters required for linguistic modeling. We conclude that **spectral density preservation** is a fundamental requirement for the efficiency of the DPI framework; the weight manifold must remain aligned with the heavy-tailed distributions characteristic of natural language.

**Ablation of Variance Calibration (20M Scale)**:
Similarly, we evaluated the necessity of Phase 6 LayerNorm calibration. In a direct 300-step evaluation, the **Non-Calibrated (No-Calib)** variant reached a loss of **6.79**, surpassing the **Calibrated** version (Loss 6.99). 

The results suggest that the high precision of the **Sequential Bootstrapping** protocol (DPI-14.1) renders post-hoc variance normalization redundant. In this regime, additional calibration steps act as signal dampeners, reducing the effective gradient conductivity of the pre-conditioned manifold.

**Table 6: Impact of Structural Preservation on Initialization Stability.**

| Configuration | Step 300 Loss | Delta vs Xavier | Manifold State |
| :--- | :--- | :--- | :--- |
| Xavier (Baseline) | 8.2057 | - | High Entropy (Noise) |
| DPI (With Calibration) | 6.9983 | -1.21 | Normalized / Dampened |
| **DPI (Structural Pure)** | **6.7964** | **-1.41** | **High-Conductivity** |

*Note: The "Structural Pure" configuration (No Whitening, No Calibration) provides the optimal balance of geometric alignment and signal flow.*

**Synthesis: The Primacy of Geometric Priors**:
The ablation data across all model scales (20M to 335M) demonstrates that the efficiency of DPI is derived from its **geometric specificity** rather than stochastic normalization. By allowing the manifold to evolve naturally from its pre-conditioned state, DPI-14.1 achieves superior convergence while minimizing empirical stabilization overhead. This confirms that the Transformer manifold is most efficient when initialized with **structured anisotropy** rather than forced statistical uniformity.
