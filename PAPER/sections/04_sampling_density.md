## 4.3 Structural and Sensitivity Investigations

### 4.3.1 Robustness to Data Sampling Density (Phase 0)

A potential criticism of DPI’s lexical seeding phase is the perceived logistical overhead of constructing co-occurrence matrices for large-scale corpora. To address this, we conducted a sensitivity analysis on sampling density over a sustained training interval of 300 steps.

#### 4.3.1.1 The Sparse Initialization Experiment
We compared two initialization regimes for a 20M parameter model to determine the minimum data requirements for Phase 0. The first regime, **Ultra-Sparse**, computed the lexical seeding on only 100 lines of raw text, while the second regime, **Standard**, used 10,000 lines.

#### 4.3.1.2 Results: Invariant Semantic Priors
The convergence trajectories were nearly identical throughout the training process, with a negligible loss delta of 0.053 at Step 300 (Ultra-Sparse: 7.06 vs. Standard: 7.01). These results prove that the macroscopic geometric structure of language is captured almost instantly, suggesting that DPI does not require processing the full training corpus for initialization.

#### 4.3.1.3 Scalability Assessment
The finding that a vanishingly small sample is sufficient to provide the structural priors required for immediate gradient conductivity has significant implications for large-scale pre-training. It effectively reduces the lexical seeding overhead to near-zero, confirming DPI’s logistical viability for industrial-scale applications.
