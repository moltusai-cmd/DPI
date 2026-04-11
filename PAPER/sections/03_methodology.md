# 3. METHODOLOGY: THE DPI FRAMEWORK

DPI replaces the standard random initialization with a deterministic, data-driven pipeline that aligns the model's weight matrices with the expected information flow of a trained Transformer.

### 3.1 Lexical Seeding (Phase 0)
Before initializing internal layers, we seed the embedding layer $E \in \mathbb{R}^{V \times d}$ by performing a Singular Value Decomposition (SVD) on a token co-occurrence matrix $C$ derived from a small data sample (100k tokens). This ensures that semantically related tokens are geometrically clustered from step zero, reducing the gradient load on the embedding space.

### 3.2 The Continuous Manifold Transition
Unlike block-based initialization, DPI implements a continuous morphing pipeline across depth $l \in [0, L]$. Each layer's weights are a mixture of syntactic and semantic bases:
$$ W_l = \omega_{syn}(l) \cdot B_{syn} + \omega_{sem}(l) \cdot B_{sem} $$
where $\omega_{syn}$ decays exponentially and $\omega_{sem}$ follows a Gaussian "Hunchback" distribution.

### 3.3 Functional Components
1.  **Syntactic Entry (DCT-II)**: We project a 2D Discrete Cosine Transform basis onto the early layers to provide frequency-aware parsing capabilities. We apply a **Zipfian Warp** ($\zeta = 1.4$) to the frequency grid to align with the power-law distribution of language.
2.  **Topological Emergence (K-Means)**: Intermediate layers are initialized using centroids from a Mini-Batch K-Means clustering of initial activations. This creates a Voronoi-partitioned latent space for concept grouping.
3.  **Semantic Core (SVD Tracking)**: We modulate the spectral density of the semantic core using a **Spectral Gamma** ($\gamma = 0.35$). The trajectory follows the **CAST Framework**, enforcing a compression bottleneck in the middle layers to drive abstraction.

### 3.4 Dynamic Isometry and Calibration
To ensure perfect gradient conductivity without warmup:
*   **QR Decomposition**: All output projection matrices ($W_O, W_2$) are strictly orthogonalized ($Q^T Q = I$).
*   **Residual Heartbeat**: We alternate residual gains ($1.2\times$ vs $0.2\times$) between odd and even layers to simulate calculation vs. storage cycles.
*   **Robust Calibration**: Final LayerNorm gains are adjusted based on the mean variance of the residual stream over multiple sample batches to maintain unit variance ($Var(x) = 1.0$) throughout the stack.
