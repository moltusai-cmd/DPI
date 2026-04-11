# 3. METHODOLOGY: THE DPI FRAMEWORK

DPI replaces stochastic initialization with a deterministic, data-driven pipeline. The framework instantiates a depth-aware weight manifold by aligning the network's initial state with the linguistic and spectral invariants of the target domain.

### 3.1 Lexical Seeding (Phase 0)
The embedding matrix $E \in \mathbb{R}^{V \times d}$ is initialized via Singular Value Decomposition (SVD) of a normalized co-occurrence matrix $C \in \mathbb{R}^{V \times V}$. Given $C = U_c \Sigma_c V_c^T$, we define:
$$ E = U_c \cdot \sqrt{\Sigma_c} $$
truncated to dimension $d$. This ensures that initial lexical distances reflect statistical proximity in the training corpus.

### 3.2 The Continuous Manifold Equation
Weights at layer $l \in [0, L-1]$ are defined as a weighted mixture of a syntactic basis $B_{syn}$ and a semantic basis $B_{sem}$:
$$ W_l = \omega_{syn}(l) \cdot B_{syn} + \omega_{sem}(l) \cdot B_{sem} $$
where the weighting functions are defined by the depth progress $p = \frac{l}{L-1}$:
*   **Syntactic decay**: $\omega_{syn}(p) = e^{-3p}$
*   **Gaussian Hunchback**: $\omega_{sem}(p) = \exp\left(-\frac{(p - 0.5)^2}{2\sigma^2}\right)$, with $\sigma=0.25$.

### 3.3 Spectral Components

#### 3.3.1 Zipfian-Warped DCT (Phase 1)
The syntactic basis $B_{syn}$ is derived from a 2D Discrete Cosine Transform (DCT-II). To align the spectral resolution with the power-law distribution of language (Zipf's Law), we apply a frequency warp $\zeta$ to the index $j$:
$$ B_{syn}(i, j) = \cos\left[ \frac{\pi}{d} \left( \left( \frac{j}{d} \right)^\zeta \cdot d + \frac{1}{2} \right) i \right] \cdot \sqrt{\frac{2}{d}} $$
where $\zeta = 1.4$ for linguistic datasets.

#### 3.3.2 CAST Spectral Modulation (Phase 3)
The semantic basis $B_{sem}$ is initialized using the SVD of initial activations. We modulate the singular values $\sigma_k$ to enforce a compression bottleneck following the CAST trajectory:
$$ B_{sem} = U \cdot \Sigma^{\gamma(p)} $$
where the modulated gamma $\gamma(p)$ follows a sinusoïdal compression profile:
$$ \gamma(p) = \gamma_0 \cdot (1 - 0.5 \sin(\pi p)) $$
with base $\gamma_0 = 0.35$.

### 3.4 Orthogonality and Calibration
To maintain dynamic isometry ($SingularValues \approx 1$) throughout the stack:
1.  **QR Decomposition**: All output projection matrices $W_{out}$ are strictly orthogonalized such that $W_{out}^T W_{out} = I$.
2.  **Variance Calibration**: LayerNorm gains $\gamma_{ln}$ are scaled by a factor $s = \mathbb{E}[\sqrt{1/Var(x)}]$ averaged over 10 independent data batches to ensure $Var(x) = 1.0$ at every layer boundary.
