### 4.1.3 Extended Convergence Analysis: Asymptotic Persistence

To investigate if stochastic initialization eventually catches up to geometric pre-conditioning, we extended the WikiText-BPE benchmark to **10 full epochs** (14,740 steps) on the 20.33M parameter architecture.

**Crossover and Time-to-Target**: The data reveals that DPI maintains a lead through the entire training cycle. As shown in the long-term trajectory (Figure 1), the initial alignment provided by geometric pre-conditioning establishes a permanent phase advantage.

\vspace{1em}
![Figure 1: Long-term Convergence Comparison. Comparison of Validation Loss between DPI-14.1 and Xavier baseline over 14,740 training steps (10 epochs) on WikiText-BPE.](figures/marathon_convergence.png){width=85%}
\vspace{1em}

While the Xavier baseline converges steadily, it fails to close the gap created by DPI's initial alignment. As illustrated by the delta persistence (Figure 2), the Xavier baseline achieves a final validation loss of 5.38 at Step 14,500, a milestone that DPI reaches significantly earlier at Step 7,000. This sustained efficiency indicates that DPI achieves in approximately 48% of the time what the standard baseline required for the entire run.

\vspace{1em}
![Figure 2: Persistence of the DPI Advantage. The loss delta ($\Delta Loss = Loss_{DPI} - Loss_{Xavier}$) remains statistically significant through the entire 10-epoch training run.](figures/marathon_delta.png){width=85%}
\vspace{1em}

**Advantage Erosion Analysis**: We observed a natural erosion of the loss delta as both models approached their theoretical capacity for the given architecture and dataset. The peak delta of -0.93 observed at Step 1,000 narrowed to -0.44 by the mid-training point (Step 4,500), eventually reaching a final delta of -0.10 at Step 14,500.

Despite this convergence, the final 0.10 delta remains statistically significant. It suggests that DPI-initialized models may reside in a more favorable local minimum, retaining a slight edge in perplexity even at full convergence.

**Evaluation of Asymptotic Convergence Persistence**: The results of this extended evaluation confirm that DPI provides more than a transient initialization benefit. It establishes a phase advantage that translates into a permanent reduction in total computational expenditure. For industrial applications where training is constrained by time or hardware resources, DPI significantly enhances the overall throughput of the pre-training process.
