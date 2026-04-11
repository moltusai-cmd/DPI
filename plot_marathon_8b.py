import json
import matplotlib.pyplot as plt
import os

def plot_8b_marathon():
    # Load DPI results
    with open("tests/Titan_8B_Survival/marathon_8b_results.json") as f:
        dpi_data = json.load(f)
    
    # Load Xavier results
    with open("tests/Titan_8B_Survival/xavier_8b_results.json") as f:
        xavier_data = json.load(f)
        
    steps_dpi = [d['step'] for d in dpi_data]
    loss_dpi = [d['loss'] for d in dpi_data]
    
    steps_xavier = [x['step'] for x in xavier_data]
    loss_xavier = [x['loss'] for x in xavier_data]
    
    plt.figure(figsize=(12, 7))
    plt.plot(steps_xavier, loss_xavier, label="Xavier Uniform (8.19B)", color='red', linestyle='--', linewidth=2)
    plt.plot(steps_dpi, loss_dpi, label="DPI (PID-14) (8.19B)", color='blue', linewidth=3)
    
    plt.title("Titan Challenge: 8-Billion Parameter Survival (0% Warmup, AdamW 8-bit)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss (ArXiv)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Highlighting the gap
    final_gap = loss_xavier[-1] - loss_dpi[-1]
    plt.annotate(f"Gap: {final_gap:.2f} Loss Points", 
                 xy=(steps_dpi[-1], loss_dpi[-1]), 
                 xytext=(steps_dpi[-1]-300, loss_dpi[-1]-0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    os.makedirs("PAPER/figures", exist_ok=True)
    plt.savefig("PAPER/figures/marathon_8b.png")
    print("Graphique 8B généré : PAPER/figures/marathon_8b.png")

if __name__ == "__main__":
    plot_8b_marathon()
