# 3. Methodology: The PID-14.1 Framework

DPI replaces stochastic initialization with a **Sequential Bootstrapping** pipeline. Unlike global initialization methods, PID-14.1 treats the network as a dynamic flow, initializing each layer using the real-time spectral signatures of the preceding manifold.

### 3.1 Lexical Seeding (Phase 0)
The embedding matrix $E \in \mathbb{R}^{V \times d}$ is initialized via a Nyström-approximated Singular Value Decomposition (SVD) of a token co-occurrence matrix. This ensures that the initial semantic space is rooted in the statistical structure of the target domain.

### 3.2 Sequential Manifold Initialization
Rather than projecting global statistics across all layers, PID-14.1 employs an iterative initialization process. For each layer $l \in [0, L-1]$:
1.  **Activation Collection**: Real activations are collected at the output of layer $l-1$ (after its own initialization).
2.  **Spectral Analysis**: The SVD of the current manifold $X_{l-1} = U_l \Sigma_l V_l^T$ is calculated to capture the signal's energy distribution at depth $l$.
3.  **Basis Mixing**: The weights are defined as a mixture of a syntactic DCT basis $B_{syn}$ and a semantic SVD basis $B_{sem}(l)$, where $B_{sem}(l) = U_l \cdot \Sigma_l^{\gamma(l)}$.

### 3.3 Differentiated Functional Signatures
To ensure head diversity and stable attention routing, PID-14.1 abandons symmetric initialization in favor of **Functional QKV Signatures**:

#### 3.3.1 The Key ($K$) Signature: Structural Orthogonality
To define distinct "axes of hypotheses," the Key projections $W_k$ are progressively orthogonalized using QR decomposition. This orthogonality peaks at the network's midpoint to maximize the search space:
$$ W_k(l) = (1 - \sin(\pi p)) \cdot M_{base} + \sin(\pi p) \cdot QR(M_{base}) $$
where $p$ is the depth progress.

#### 3.3.2 The Value ($V$) Signature: Manifold Deployment
The Value projections $W_v$ are constrained to a low-rank variety by applying pronounced spectral compression ($\gamma_v \approx 0.4 \gamma_{base}$). This forces the model to encode information along dominant principal components (PC1 dominance), facilitating stable value propagation.

#### 3.3.3 The Query ($Q$) Signature: Routing Alignment
The Query projections $W_q$ are initially aligned with the Keys to bootstrap basic attention mechanisms, then gradually diverge toward independent routing axes as depth increases, allowing for complex cross-layer dependencies.

### 3.4 Isometry and Calibration
To prevent gradient instability at billion-parameter scales:
1.  **Strict Orthogonality**: All output and MLP projection matrices are initialized via QR decomposition to ensure $W^T W = I$.
2.  **Dynamic Isometry (Phase 6)**: LayerNorm gains are calibrated by measuring the empirical variance of the sequential signal flow, ensuring $Var(x) \approx 1.0$ at every manifold boundary.
