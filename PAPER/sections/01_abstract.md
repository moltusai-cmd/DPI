# ABSTRACT

Standard stochastic initialization methods (Xavier, He) are geometrically blind to the structure of natural language, forcing Transformer architectures to spend thousands of gradient steps recovering from anisotropic collapse before meaningful semantic learning can begin. This "warmup tax" is not a mathematical necessity — it is an artifact of treating the network as a blank slate.

We introduce **DPI (Deterministic Pipeline Initialization)**, a depth-aware pre-conditioning framework that replaces stochastic noise with targeted geometric structures derived from the data itself. DPI applies a continuous morphing pipeline across network depth: early layers receive DCT-based frequency priors aligned with lexical distribution, intermediate layers are initialized via K-Means and SVD decompositions of real activation statistics, and all layers are finalized through QR orthogonalization and variance calibration. This instantiates a stable, data-aware latent manifold at step zero without requiring any warmup schedule.

We evaluate DPI against Xavier initialization on Transformer models ranging from 20M to 300M parameters, across multiple datasets (WikiText-103, arXiv abstracts) and tokenization schemes (BPE, word-level). DPI consistently reaches lower loss at every measured checkpoint, and maintains its advantage through full training — converging to a perplexity **1.65x lower** than Xavier on equivalent compute. The gap does not close with additional epochs, suggesting DPI reaches a structurally distinct and more favorable loss basin.

These results suggest that initialization is not a neutral starting condition, but a geometric prior with lasting consequences on the optimization landscape.
