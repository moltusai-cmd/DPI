import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_marathon():
    # Load data from the archived marathon test
    test_dir = "tests/Marathon_10Epoch_20M"
    dpi_df = load_data(f"{test_dir}/duel_20m_dpi_nowhite.json")
    xav_df = load_data(f"{test_dir}/duel_20m_xavier.json")
    
    dpi_val = dpi_df[dpi_df['val_loss'].notna()]
    xav_val = xav_df[xav_df['val_loss'].notna()]
    
    os.makedirs("PAPER/figures", exist_ok=True)
    
    # 1. LONG-TERM CONVERGENCE
    plt.figure(figsize=(12, 7))
    plt.plot(xav_val['step'], xav_val['val_loss'], label='Xavier (Standard)', color='red', alpha=0.7)
    plt.plot(dpi_val['step'], dpi_val['val_loss'], label='DPI No-White (PID-14)', color='blue', linewidth=2)
    
    plt.title("Long-Term Convergence: The 10-Epoch Marathon (20M Params)")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Validation Loss")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig("PAPER/figures/marathon_convergence.png")
    print("Saved: PAPER/figures/marathon_convergence.png")
    
    # 2. DELTA EROSION PLOT
    merged = pd.merge(xav_val[['step', 'val_loss']], dpi_val[['step', 'val_loss']], on='step', suffixes=('_xav', '_dpi'))
    merged['delta'] = merged['val_loss_xav'] - merged['val_loss_dpi']
    
    plt.figure(figsize=(12, 7))
    plt.plot(merged['step'], merged['delta'], color='purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.fill_between(merged['step'], 0, merged['delta'], color='purple', alpha=0.1)
    
    plt.title("Advantage Persistence (Xavier - DPI)")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Loss Delta")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig("PAPER/figures/marathon_delta.png")
    print("Saved: PAPER/figures/marathon_delta.png")

if __name__ == "__main__":
    plot_marathon()
