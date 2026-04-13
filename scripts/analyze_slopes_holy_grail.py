import json
import numpy as np

def analyze_slopes_at_targets(json_path, targets=[6.0, 5.0, 4.5, 4.0, 3.7, 3.5]):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xavier = np.array(data['xavier'])
    dpi = np.array(data['dpi'])
    
    def get_instant_slope(history, target):
        # On cherche le premier point où on descend sous le target
        indices = np.where(history <= target)[0]
        if len(indices) == 0:
            return None, None
        
        idx = indices[0]
        if idx < 100:
            return None, None
            
        # On calcule la pente sur une fenêtre de 100 steps autour du point
        # Pente = (L_fin - L_debut) / (Steps)
        window = 100
        start_idx = max(0, idx - window // 2)
        end_idx = min(len(history) - 1, idx + window // 2)
        
        slope = (history[start_idx] - history[end_idx]) / (end_idx - start_idx)
        return slope, idx + 1

    print(f"{'Target Loss':<12} | {'Step Xavier':<12} | {'Slope X':<12} | {'Step DPI':<12} | {'Slope D':<12} | {'Factor'}")
    print("-" * 85)
    
    for t in targets:
        slope_x, step_x = get_instant_slope(xavier, t)
        slope_d, step_d = get_instant_slope(dpi, t)
        
        if slope_x and slope_d:
            factor = slope_d / slope_x
            print(f"{t:<12.2f} | {step_x:<12} | {slope_x:<12.6f} | {step_d:<12} | {slope_d:<12.6f} | {factor:<12.2f}x")
        else:
            status = "Target not reached by both"
            print(f"{t:<12.2f} | {status}")

if __name__ == "__main__":
    analyze_slopes_at_targets("marathon_100k_holy_grail.json")
