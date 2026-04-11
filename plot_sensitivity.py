import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_sensitivity():
    # Load the FINAL triangulation data (1 Epoch)
    with open("tests/Triangulation_Final_1Epoch/triangulation_1epoch.json") as f:
        data = json.load(f)
    
    df = pd.DataFrame([
        {**d['params'], 'loss': d['val_loss']} for d in data
    ])
    
    os.makedirs("PAPER/figures", exist_ok=True)
    
    # 1. SENSITIVITY TO ZIPF WARP (zeta)
    plt.figure(figsize=(10, 6))
    zw_stats = df.groupby('zipf_warp')['loss'].agg(['mean', 'std']).reset_index()
    plt.errorbar(zw_stats['zipf_warp'], zw_stats['mean'], yerr=zw_stats['std'], fmt='o-', capsize=5, color='blue')
    plt.title("Sensitivity Analysis: Zipfian Warp ($\zeta$)")
    plt.xlabel("Warp Factor ($\zeta$)")
    plt.ylabel("Validation Loss (1 Epoch)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("PAPER/figures/sensitivity_zeta.png")
    
    # 2. SENSITIVITY TO SPECTRAL GAMMA (gamma)
    plt.figure(figsize=(10, 6))
    sg_stats = df.groupby('spectral_gamma')['loss'].agg(['mean', 'std']).reset_index()
    plt.errorbar(sg_stats['spectral_gamma'], sg_stats['mean'], yerr=sg_stats['std'], fmt='o-', capsize=5, color='red')
    plt.title("Sensitivity Analysis: Spectral Gamma ($\gamma_0$)")
    plt.xlabel("Gamma Constant ($\gamma_0$)")
    plt.ylabel("Validation Loss (1 Epoch)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("PAPER/figures/sensitivity_gamma.png")

    print("Sensitivity plots generated in PAPER/figures/")

if __name__ == "__main__":
    plot_sensitivity()
