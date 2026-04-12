import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_sprint():
    # Load data from the archived sprint test
    dpi_df = load_data("tests/Sprint_5Epoch/duel_20m_dpi_nowhite.json")
    xav_df = load_data("tests/Sprint_5Epoch/duel_20m_xavier.json")
    
    # Filter only rows where val_loss exists for a clean plot
    dpi_val = dpi_df[dpi_df['val_loss'].notna()]
    xav_val = xav_df[xav_df['val_loss'].notna()]
    
    os.makedirs("PAPER/figures", exist_ok=True)
    
    # 1. CONVERGENCE PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(xav_val['step'], xav_val['val_loss'], label='Xavier (Standard)', color='red', linestyle='--', marker='o')
    plt.plot(dpi_val['step'], dpi_val['val_loss'], label='DPI No-White (PID-14)', color='blue', marker='s')
    
    plt.title("Transformer 20M: Convergence Comparison (WikiText-BPE)")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Validation Loss")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig("PAPER/figures/convergence_sprint.png")
    print("Saved: PAPER/figures/convergence_sprint.png")
    
    # 2. DELTA PLOT (Xavier - DPI)
    # We need to align steps
    merged = pd.merge(xav_val[['step', 'val_loss']], dpi_val[['step', 'val_loss']], on='step', suffixes=('_xav', '_dpi'))
    merged['delta'] = merged['val_loss_xav'] - merged['val_loss_dpi']
    
    plt.figure(figsize=(10, 6))
    plt.plot(merged['step'], merged['delta'], color='green', linewidth=2, marker='^')
    plt.axhline(y=0, color='black', linestyle='-')
    
    plt.title("Performance Advantage (Xavier Loss - DPI Loss)")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Loss Delta (Higher is Better for DPI)")
    plt.fill_between(merged['step'], 0, merged['delta'], color='green', alpha=0.2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig("PAPER/figures/delta_advantage.png")
    print("Saved: PAPER/figures/delta_advantage.png")

if __name__ == "__main__":
    plot_sprint()
