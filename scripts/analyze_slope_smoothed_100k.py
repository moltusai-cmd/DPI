import json
import numpy as np
from collections import defaultdict

def analyze_slope_smoothed_100k(json_path, bin_size=0.05):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xavier = np.array(data['xavier'])
    dpi = np.array(data['dpi'])
    
    def get_bin_stats(history):
        # Calcul des pentes locales : L(t-1) - L(t) 
        # (Positif si la loss descend)
        slopes = history[:-1] - history[1:]
        losses = history[1:]
        
        bins = defaultdict(list)
        for loss, slope in zip(losses, slopes):
            # On arrondit au bin de 0.05 le plus proche
            b = round(round(loss / bin_size) * bin_size, 2)
            bins[b].append(slope)
        
        # Calcul des moyennes par bin
        stats = {b: np.mean(v) for b, v in bins.items() if len(v) > 0}
        counts = {b: len(v) for b, v in bins.items()}
        return stats, counts

    print("Computing smoothed statistics...")
    stats_x, counts_x = get_bin_stats(xavier)
    stats_d, counts_d = get_bin_stats(dpi)
    
    print("\n" + "="*85)
    print(f"{'Loss Bin':<10} | {'Slope X (avg)':<15} | {'Slope D (avg)':<15} | {'Advantage':<12} | {'Samples (X/D)'}")
    print("-" * 85)
    
    # On trie les bins de la plus haute loss vers la plus basse
    all_bins = sorted(set(stats_x.keys()).intersection(set(stats_d.keys())), reverse=True)
    
    for b in all_bins:
        sx = stats_x[b]
        sd = stats_d[b]
        cx = counts_x[b]
        cd = counts_d[b]
        
        if sx > 0:
            ratio = sd / sx
            print(f"{b:<10.2f} | {sx:<15.7f} | {sd:<15.7f} | {ratio:<12.2f}x | {cx:<5}/{cd:<5}")
        else:
            print(f"{b:<10.2f} | {sx:<15.7f} | {sd:<15.7f} | {'N/A':<12} | {cx:<5}/{cd:<5}")

if __name__ == "__main__":
    analyze_slope_smoothed_100k("results/marathon_100k_holy_grail.json")
