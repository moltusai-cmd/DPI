### 4.1.2 Quantitative ROI and Qualitative Syntax Analysis

To provide a precise economic justification for DPI, we measured the **Compute Return on Investment (ROI)** across different performance thresholds.

#### 4.1.2.1 Compute ROI Analysis (20M Scale)
The following table tracks the number of training steps required to reach specific Validation Loss targets (Table 2).

Table 2: Compute Return on Investment (ROI) and Step-Efficiency on WikiText-BPE.

| Target Loss | Xavier Steps | DPI Steps | Compute ROI (Multiplier) |
| :--- | :--- | :--- | :--- |
| **8.5** (Initial Syntax) | 450 | 45 | **10.0x** |
| **7.5** (Pattern Discovery) | 900 | 180 | **5.0x** |
| **6.5** (Semantic Alignment) | 1,600 | 350 | **4.57x** |
| **6.2** (Base Convergence) | 8,000 | 1,600 | **5.0x** |

**Conclusion**: DPI delivers a sustained **4.6x to 5.0x compute saving** across the entire training duration. To achieve the same quality as a 1-epoch DPI run, a stochastic model requires nearly 5 epochs of training.

#### 4.1.2.2 Qualitative Syntax Maturity
We analyzed the early-step output of both models to identify the "Maturity Gap":
1.  **Xavier @ Step 100**: "the . the , of and the . . ." (Repetitive token sequences).
2.  **DPI @ Step 100**: "the species of the forest , which was discovered by the . . ." (Structured noun phrases and clausal dependencies).

DPI models skip the "punctuation learning" phase entirely, entering the "relational learning" phase from the first update. This explains why the loss advantage is so substantial in the first 500 steps.
