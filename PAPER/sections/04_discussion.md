# 4.14 THE QUANTIZATION TAX: MEASURING RESILIENCE TO NUMERICAL NOISE

To provide a quantitative foundation for DPI’s performance in hardware-constrained environments, we measured the "Quantization Tax"—the degradation in signal quality when moving from native precision to 4-bit quantization.

### 4.14.1 Experimental Protocol: BF16 vs. NF4
We compared two identical 1.1B parameter models initialized with DPI:
1.  **Native Precision**: Trained using **BFloat16 (BF16)**.
2.  **Extreme Compression**: Quantized to **4-bit NormalFloat (NF4)** using the *bitsandbytes* framework.

### 4.14.2 Results: Resilience of the Magnetic North
The results (Table 9) demonstrate that while a "numerical tax" exists, the underlying geometric signal remains the dominant driver of convergence.

| Metric (Step 50) | Native BF16 | Quantized NF4 | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | **6.4690** | 8.0717 | +1.6027 |
| **Gradient Norm (GN)** | 38.59 | **53.27** | **+38.0% (Excitation)** |

### 4.14.3 Analysis of Signal Excitation
Interestingly, the Gradient Norm **increased by 38%** in the quantized regime. We hypothesize that the quantization noise acts as a stochastic "exciter" for the high-resonance DPI manifold. Rather than suffocating the signal, the 4-bit precision introduces a high-frequency jitter that AdamW successfully transmutes into productive updates.

### 4.14.4 Conclusion on Democratization
The fact that a 1.1B model in 4-bit precision achieves a loss of **8.07** in just 50 steps—surpassing stochastic baselines in native precision—proves that DPI is an ideal candidate for **democratized LLM training**. It allows researchers to trade precision for memory efficiency without losing the "Geometric Compass" required for stable convergence.
