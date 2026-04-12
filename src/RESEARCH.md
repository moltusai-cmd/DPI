# DPI Research & Development Notes

## The v17.x Population Experiments (The "Head Mystery")

During the development of DPI v16.2, we investigated whether all attention heads should be initialized with the same alignment prior, or if a "population-based" approach was superior.

### v17.0: Entropy Spreading
- **Concept**: Diversify heads into three groups: Exploratory (5%), Standard (40%), and Focal (80%).
- **Result**: Significant degradation in early loss. The model struggled to establish a coherent semantic manifold because high-variance SVD components were assigned to exploratory heads.

### v17.1: Principal Focalization (The "Mystery" Version)
- **Concept**: Align head specialization with SVD eigenvalues (Principal Component Hierarchy).
    - **Top 25% Heads**: Focal (0.80) - Anchoring the dominant signal.
    - **Mid 50% Heads**: Standard (Genomic Peak).
    - **Bottom 25% Heads**: Exploratory (0.05) - Letting residual noise wander.
- **Result**: Achieved the **lowest Step 1 Validation Loss in history (9.04)**.
- **Conclusion**: While v17.1 is the "fastest starter," it creates geometric constraints that the Adam optimizer must eventually resolve. By Step 200, the **Uniform Alignment (v16.2)** systematically takes the lead.

### Why v16.2 (Uniform) Wins
The optimization landscape of Transformers favors symmetry at initialization. Forcing specialization "pre-pays" a debt that the model would rather settle naturally through gradient descent. v16.2 provides the perfect "consensus" starting point.

---
*For production, always use v16.2. For research into head specialization, v17.1 remains the state-of-the-art reference for lexical anchoring.*
