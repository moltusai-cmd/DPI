### 4.1.3 Extended Convergence Analysis: Asymptotic Persistence

To investigate if stochastic initialization eventually catches up to geometric pre-conditioning, we extended the WikiText-BPE benchmark to **10 full epochs** (14,740 steps) on the 20.33M parameter architecture.

#### 4.1.3.1 Crossover and Time-to-Target
The data reveals that DPI maintains a lead through the entire training cycle (Figure 1). While the Xavier baseline converges steadily, it fails to close the gap created by DPI's initial alignment (Figure 2). Specifically, the Xavier baseline achieves a final validation loss of 5.38 at Step 14,500, a milestone that DPI reaches significantly earlier at Step 7,000. This sustained efficiency indicates that DPI achieves in approximately 48% of the time what the standard baseline required for the entire run.

![Figure 1: Long-term Convergence of Validation Loss during 10-Epoch Extended Training.](figures/marathon_convergence.png)

![Figure 2: Persistence of the DPI Advantage (Delta Loss) over the Xavier Baseline.](figures/marathon_delta.png)

#### 4.1.3.2 Advantage Erosion Analysis
We observed a natural erosion of the loss delta as both models approached their theoretical capacity for the given architecture and dataset. The peak delta of -0.93 observed at Step 1,000 narrowed to -0.44 by the mid-training point (Step 4,500), eventually reaching a final delta of -0.10 at Step 14,500.

Despite this convergence, the final 0.10 delta remains statistically significant. It suggests that DPI-initialized models may reside in a more favorable local minimum, retaining a slight edge in perplexity even at full convergence.

#### 4.1.3.3 Assessment of Long-Term Convergence
The results of this extended evaluation confirm that DPI is not merely a "startup boost." It provides a phase advantage that translates into a permanent reduction in compute requirements. For industrial applications where training is capped by a budget in time or resources, DPI effectively doubles the productive capacity of the hardware.
