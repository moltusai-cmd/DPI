# HOWTO: Working with Deterministic Pipeline Initialization (DPI)

This guide provides practical instructions for reproducing our research results and integrating DPI-14.1 into your own projects.

## 🛠 1. Environment Setup

DPI-14.1 requires Python 3.10+ and a CUDA-enabled environment for optimal performance.

```bash
# Clone the repository
git clone https://github.com/your-username/DPI.git
cd DPI

# Install dependencies
pip install torch scikit-learn tokenizers
```

## 🚀 2. Reproducing the "Quick Duel"

To verify the DPI advantage on your hardware, run the 100-step 20M parameter duel. This script compares DPI-14.1 against a Xavier baseline in real-time.

```bash
python3 scripts/quick_duel.py
```

**What to expect:**
- **Phase 0-3 logs**: You will see the sequential bootstrapping of layers.
- **Convergence gap**: Within 100 steps, DPI typically shows a >1.0 point loss advantage over stochastic methods.

## 🏗 3. Integrating DPI into Your Project

To use DPI-14.1 in your own Transformer implementation, follow these steps:

### Step 1: Core Imports
Ensure your model architecture is compatible (DPI expects a standard Decoder-only Transformer structure with `nn.ModuleList` named `layers`).

```python
import sys
import os
sys.path.append('path/to/DPI/src')

from initialize_dpi import initialize_dpi
```

### Step 2: Prepare a Sample DataLoader
DPI is data-aware. It needs a small sample of your training data (e.g., 100-500 batches) to analyze the latent manifold.

```python
# A standard PyTorch DataLoader providing [Batch, Seq_len] tensors
sample_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
```

### Step 3: Initialize the Manifold
Apply the bootstrapping before you start your training loop.

```python
model = YourTransformer(...) # Must be on the target device (e.g., .to('cuda'))

initialize_dpi(
    model, 
    sample_loader,
    warp_zeta=1.1,       # Zipfian spectral warp
    spectral_gamma=0.25, # SVD compression factor
    use_calibration=True # Apply Phase 6 LayerNorm scaling
)
```

## 🧪 4. Running the 8.19B Scaling Test

For large-scale validation (8B+ parameters), we recommend using **4-bit NF4 quantization** to fit the model on a single 16GB/24GB GPU.

1.  Navigate to the `experiments/Titan_8B_Survival/` directory.
2.  Review `scripts/run_8b_duel.py` for specific hyperparameter settings (Virtual Batch Size = 32).
3.  Execute using the **S-DPI Hybrid** configuration for maximum stability at high learning rates.

## 📈 5. Visualizing Results

The `scripts/` folder contains plotting utilities to analyze your training runs:

```bash
# Plot the marathon (10-epoch) convergence curve
python3 scripts/plot_marathon.py --results results/duel_20m_dpi.json
```

## ❓ Troubleshooting

- **NaNs during Init**: Ensure your `sample_loader` provides valid token IDs. If using extremely small models (<10M), reduce `num_samples` in `get_activations`.
- **Memory Errors**: DPI initialization requires a temporary activation buffer. If OOM occurs, reduce the `batch_size` of your `sample_loader`.
- **DPI Stagnation**: If DPI does not show an advantage, check your `spectral_gamma`. Values between 0.15 and 0.45 are recommended for most natural language corpora.
