# GEMINI Project Mandates: DPI Framework

This document defines the foundational mandates and architectural constraints for the Deterministic Pipeline Initialization (DPI) project. These instructions take absolute precedence over general workflows.

## 🎯 Core Objective
Establish **DPI v16.2 "Phase-Shift Genomic"** as the state-of-the-art framework for Transformer initialization, maintaining a **1.21 point validation loss advantage** over Xavier baselines and a **5x compute efficiency multiplier**.

## ⚖️ Architectural Mandates (The Gold Standard)

### 1. Initialization Protocol
- **Default Mode**: Always use `initialize_dpi(model, dataloader, mode="v16.2")`.
- **Phase 0 (Lexical)**: SVD-based embedding seeding is mandatory.
- **Phase 2 (Genomic)**: 
    - **Phase-Shift**: Transition from "Exploratory" ($K \neq V$) to "Consolidated" ($K=V$) at ~42% depth.
    - **QK-Alignment**: Peak at 0.40 in exploratory layers; near-zero (0.0001) in consolidated layers.
- **Phase 4 (Zero-Wait Head)**: Mandatory output head calibration using the inverse lexical manifold.

### 2. Hyperparameter Constraints
- **Warmup**: Strictly **0% to 0.5%** for DPI models. Forcing higher warmup sabotages the initial manifold advantage.
- **Jitter**: MLP Jitter must be fixed at **0.02**. Values > 0.04 degrade the geometric manifold.
- **Learning Rate**: Target $10^{-4}$ for AdamW on standard 20M-100M scales.

### 3. Critical Dependencies
- **Tokenizer**: Absolute consistency between the `sample_loader` used for DPI and the training data. Mismatches result in semantic noise.
- **Structural Invariants**: Follow Gemma-35B constants ($K=V$ symmetry, compression bias) in late-stage layers.

## 🔬 Research & Validation Standards
- **Baseline Comparison**: Every major architectural change MUST be benchmarked against a **Xavier Uniform** baseline (with 2% warmup) for at least 1,000 steps.
- **The v17.x Prohibition**: Do NOT use "Entropy Spreading" or "Principal Focalization" (v17.x) for production releases. **v16.2 Uniform Alignment** is the verified global optimum.
- **DNA Testing**: Use `scripts/test_dna_universality_real.py` to verify symbolic specificity if the underlying vocabulary changes.

## 📂 Core Artifacts
- **Engine**: `src/initialize_dpi.py`
- **Architecture**: `src/model.py`
- **Reference**: `PAPER/DPI_Research_Paper.pdf`
- **Mystery Archive**: `src/RESEARCH.md` (Explains why v16.2 beats v17.x).

## 🏛️ Scientific Rigor & Academic Integrity

To ensure maximum credibility within the global research community (ArXiv, ICLR, NeurIPS), all project artifacts must adhere to the highest standards of academic discourse.

- **Terminology Prohibition**: Strictly forbid the use of "marketing" or "hype-based" terminology in public documentation, commit messages, or research papers. 
    - *Prohibited*: "Holy Grail", "Supercharged", "Battle/Duel", "Money Table/Plot", "Magic/Miracle", "Sonic Regime".
    - *Required*: "Long-term Convergence Analysis", "Geometrically Augmented", "Comparative Analysis", "Principal Performance Summary", "Empirical Robustness", "High-Stability Regime".
- **Evidence-Based Claims**: All performance claims (e.g., "7.1x speedup") must be supported by reproducible logs (`results/*.json`) and standardized benchmark protocols (C3, Pareto-Optimal).
- **Tone Mandate**: Maintain a neutral, objective, and analytical tone. Focus on geometric properties, spectral integrity, and manifold conductivity rather than competitive dominance.
- **Documentation Parity**: Ensure absolute numerical consistency between `README.md`, `PAPER/*.md`, and experimental results.

---
*DPI is a "Symbolic/Semantic Manifold Accelerator." Prioritize geometric alignment over stochastic exploration.*
