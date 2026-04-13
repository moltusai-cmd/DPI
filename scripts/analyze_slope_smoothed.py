import json
import numpy as np
from collections import defaultdict

def analyze_slope_smoothed(json_path, loss_start=8.5, loss_end=3.8, bin_size=0.05):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    def get_all_slopes(history):
        slopes = []
        for i in range(1, len(history)):
            step_diff = history[i]['step'] - history[i-1]['step']
            loss_diff = history[i-1]['loss'] - history[i]['loss']
            if step_diff > 0:
                # On stocke (loss_moyenne_du_segment, pente)
                avg_loss = (history[i]['loss'] + history[i-1]['loss']) / 2
                slope = loss_diff / step_diff
                slopes.append((avg_loss, slope))
        return slopes

    slopes_x = get_all_slopes(data['xavier'])
    slopes_d = get_all_slopes(data['dpi'])
    
    # Groupement par bins
    bins_x = defaultdict(list)
    bins_d = defaultdict(list)
    
    for loss, slope in slopes_x:
        b = round(round(loss / bin_size) * bin_size, 2)
        bins_x[b].append(slope)
        
    for loss, slope in slopes_d:
        b = round(round(loss / bin_size) * bin_size, 2)
        bins_d[b].append(slope)
        
    print(f"{'Loss Bin':<12} | {'Mean Slope X':<15} | {'Mean Slope D':<15} | {'Advantage (x)'}")
    print("-" * 70)
    
    # On trie les bins par ordre décroissant (début d'entraînement -> fin)
    all_bins = sorted(set(list(bins_x.keys()) + list(bins_d.keys())), reverse=True)
    
    for b in all_bins:
        if b in bins_x and b in bins_d:
            m_x = np.mean(bins_x[b])
            m_d = np.mean(bins_d[b])
            if m_x > 0:
                ratio = m_d / m_x
                print(f"{b:<12.2f} | {m_x:<15.6f} | {m_d:<15.6f} | {ratio:<15.2f}x")

if __name__ == "__main__":
    analyze_slope_smoothed("marathon_20k_results.json")
