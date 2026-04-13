### 4.2.1 Comparison with Standard Stochastic Baselines (20M Scale)

The performance of the DPI framework was evaluated in a head-to-head comparison against the standard Xavier Uniform initialization baseline. 

**Experimental Protocol**: All tests were performed on a 20.33M parameter Transformer using the WikiText-BPE corpus. The Xavier baseline utilized a 2% warmup (140 steps) and gradient clipping. For DPI, we evaluated the **v16.2 (Structural Manifold Alignment)** configuration: Sequential Bootstrapping with a **Depth-Dependent Transition** at $L/2$ where $K$ and $V$ are consolidated, the application of "Warm Signal" variance calibration, and **Deterministic Output Head Calibration (DOHC)** (Phase 4).

**Table 3: Convergence metrics for DPI v16.2 vs. Xavier Baseline (20M Parameters).**

| Milestone (Step) | Xavier Loss | DPI v16.2 Loss | Delta (Loss) |
| :--- | :--- | :--- | :--- |
| **Step 1** | 9.7040 | **8.0378** | **-1.66** |
| **Step 50** | 9.4219 | **7.4120** | **-2.01** |
| **Step 250** | 8.6534 | **6.1082** | **-2.54** |
| **Step 1,000** | 7.1103 | **5.8950** | **-1.21** |

**Analysis of Results**:

1.  **Immediate Grammatical Coherence**: By calibrating the output head with the inverse lexical manifold (DOHC, Phase 4), DPI v16.2 achieves a **1.66 point loss advantage** at Step 1. The model demonstrates structural coherence and initial perplexity reduction prior to the first stochastic update.

2.  **Peak Divergence at Step 250**: The performance gap widens significantly in the early training phase, reaching its maximum delta of **2.54 points** at Step 250. This confirms that geometric pre-conditioning provides a fundamentally more conductive gradient path than stochastic noise.

3.  **End-to-End Manifold Alignment**: The combination of structural layer-wise transitions and deterministic head calibration ensures that information flow is optimized from input embeddings through the internal Transformer blocks to the final classification layer, maximizing the efficiency of the initial training budget.
