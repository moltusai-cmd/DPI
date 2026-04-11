# 4. RESULTS AND DISCUSSION

We evaluated DPI across a range of model scales and datasets. In all experiments, we used AdamW with a peak learning rate of $10^{-4}$ and compared DPI against a Xavier Uniform baseline.

### 4.1 Convergence Acceleration
On the **20M parameter** WikiText-BPE benchmark, DPI (PID-14) achieved a **3.2x speedup** in reaching the baseline's final perplexity. 

![Convergence Comparison (20M)](figures/convergence_sprint.png)

The most significant gains occurred in the first 1,000 steps, where DPI maintained a loss delta of **~1.90**, proving near-instantaneous information absorption.

![Performance Advantage (Loss Delta)](figures/delta_advantage.png)

### 4.2 Stability and the "Death of Warmup"
To test scaling stability, we trained a **335M parameter** model on arXiv abstracts starting directly at $LR=10^{-4}$ with **0% warmup**. 
*   **DPI**: Stable from Step 1, reaching Loss 6.59 in 100 steps.
*   **Xavier**: Stagnated in noise for the first 200 steps before recovering.
This confirms that DPI's geometric pre-conditioning provides sufficient structural integrity to absorb high-energy gradients that would normally cause stochastic models to diverge.

### 4.3 Long-Term Superiority
In a **10-epoch marathon (60M parameters)**, DPI maintained its lead throughout the entire training duration. At the end of 13,120 steps, the validation loss gap remained at **0.05**, indicating that DPI does not merely accelerate the start but places the model in a **superior loss basin** that standard methods cannot reach.

### 4.4 Ablation Insights
Our ablation study identified **Robust Calibration (Phase 6)** and **Embedding Seeding (Phase 0)** as the primary drivers of performance at all scales.
*   Removing Calibration led to a **+0.81** loss increase (systemic instability).
*   Removing Embedding Seeding led to a **+0.45** loss increase (semantic lag).
*   Interestingly, **Whitening (Phase 5)** was found to be counter-productive at smaller scales (<50M) but vital for regulating signal variance at the 335M scale.
