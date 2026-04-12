### 4.1.2 Quantitative Efficiency and Qualitative Syntax Analysis

To provide a precise technical justification for DPI, we measured the **Relative Compute Efficiency** across different performance thresholds.

**Compute Efficiency Analysis (20M Scale)**: The following table tracks the number of training steps required to reach specific Validation Loss targets (Table 2).

**Table 2: Relative Compute Efficiency and Step-Efficiency on WikiText-BPE.**

| Target Loss | Xavier Steps | DPI Steps | Efficiency Multiplier |
| :--- | :--- | :--- | :--- |
| **8.5** (Initial Syntax) | 450 | 45 | **10.0x** |
| **7.5** (Pattern Discovery) | 900 | 180 | **5.0x** |
| **6.5** (Semantic Alignment) | 1,865 | 564 | **3.31x** |
| **6.2** (Base Convergence) | 8,000 | 1,600 | **5.0x** |

*Note: The efficiency multiplier is calculated as the ratio of Xavier steps to DPI steps required to reach the target loss. DPI consistently delivers a 3.3x to 10.0x step-wise speedup.*

**Wall-Clock Efficiency (End-to-End ROI)**: To address the computational overhead of DPI’s analytical phases, we conducted a "Wall-Clock" benchmark on an NVIDIA RTX 5080. We measured the total time ($T_{total}$) required to reach a semantic alignment threshold of **Loss = 6.5**, including the initialization cost (Table 3).

**Table 3: End-to-End Wall-Clock Efficiency (RTX 5080, Batch 32).**

| Method | $T_{init}$ (s) | $T_{step}$ (s) | Steps $\rightarrow$ 6.5 | $T_{total}$ (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Xavier Baseline** | **0.001** | **0.0229** | 1,865 | 42.75 |
| **DPI-14.1** | 2.372 | 0.0237 | **564** | **15.74** |

*Note: $T_{total} = T_{init} + (T_{step} \times Steps)$. Despite a 2,372x higher initialization cost, DPI is **2.71x faster** end-to-end to reach the target loss. The initial 2.37s "investment" in geometric pre-conditioning is recovered over 11 times during the first 1,000 training steps.*

**Qualitative Evaluation of Early Syntactic Maturity**: We analyzed the early-step output of both models to identify the "Maturity Gap":
1.  **Xavier @ Step 100**: "the . the , of and the . . ." (Repetitive token sequences).
2.  **DPI @ Step 100**: "the species of the forest , which was discovered by the . . ." (Structured noun phrases and clausal dependencies).

DPI models skip the "punctuation learning" phase entirely, entering the "relational learning" phase from the first update. This explains why the loss advantage is so substantial in the first 500 steps.
