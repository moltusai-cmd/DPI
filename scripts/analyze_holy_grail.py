import json
import numpy as np

def analyze_holy_grail(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xavier = np.array(data['xavier'])
    dpi = np.array(data['dpi'])
    
    # 1. Calcul des Milestones de Loss
    print(f"{'Milestone (Step)':<18} | {'Xavier Loss':<15} | {'DPI Loss':<15} | {'Delta (Loss)':<12}")
    print("-" * 65)
    for m in [1000, 10000, 25000, 50000, 75000, 100000]:
        idx = m - 1
        lx, ld = xavier[idx], dpi[idx]
        print(f"{m:<18} | {lx:<15.4f} | {ld:<15.4f} | {lx-ld:<12.4f}")

    # 2. Calcul du Time-to-Target (Avantage de Compute)
    # On lisse un peu la courbe pour éviter les oscillations locales
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    xs = smooth(xavier, 500)
    ds = smooth(dpi, 500)

    print("\n" + "="*75)
    print("🏆 THE HOLY GRAIL COMPUTE EFFICIENCY (Xavier Steps / DPI Steps)")
    print("="*75)
    print(f"{'Target Loss':<15} | {'Xavier Step':<15} | {'DPI Step':<15} | {'Efficiency Boost'}")
    print("-" * 75)
    
    # On cherche l'avantage pour des paliers de loss de plus en plus bas
    for target in [5.5, 5.0, 4.5, 4.0, 3.7, 3.5]:
        # Trouver le premier index où la courbe lissée descend sous le target
        idx_x = np.where(xs <= target)[0]
        idx_d = np.where(ds <= target)[0]
        
        if len(idx_x) > 0 and len(idx_d) > 0:
            s_x = idx_x[0] + 1
            s_d = idx_d[0] + 1
            boost = s_x / s_d
            print(f"{target:<15.2f} | {s_x:<15} | {s_d:<15} | {boost:<15.2f}x")

    # 3. Conclusion sur le Bassin de Convergence
    final_gain = np.exp(xavier[-1]) / np.exp(dpi[-1])
    print("\n" + "="*75)
    print(f"FINAL PERPLEXITY ADVANTAGE: {final_gain:.2f}x lower perplexity for DPI")
    print("="*75)

if __name__ == "__main__":
    analyze_holy_grail("marathon_100k_holy_grail.json")
