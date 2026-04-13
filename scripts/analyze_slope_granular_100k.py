import json
import numpy as np

def analyze_slope_granular_100k(json_path, step_size=0.1):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    x_loss = np.array(data['xavier'])
    d_loss = np.array(data['dpi'])
    
    # Slopes (dL/dt)
    x_slopes = x_loss[:-1] - x_loss[1:]
    d_slopes = d_loss[:-1] - d_loss[1:]
    
    # We want to map loss to slope
    # Start from max loss down to min loss
    max_l = max(np.max(x_loss), np.max(d_loss))
    min_l = min(np.min(x_loss), np.min(d_loss))
    
    # Standardize to 1 decimal place steps
    targets = np.arange(np.floor(max_l * 10) / 10, np.ceil(min_l * 10) / 10, -step_size)
    
    print(f"\n{'Target Loss':<12} | {'Slope X':<12} | {'Slope D':<12} | {'Advantage'}")
    print("-" * 60)
    
    for t in targets:
        # Find index where loss first crosses target t
        idx_x = np.where(x_loss <= t)[0]
        idx_d = np.where(d_loss <= t)[0]
        
        if len(idx_x) > 0 and len(idx_d) > 0:
            ix = idx_x[0]
            id = idx_d[0]
            
            # Check if index is within slope range
            if ix < len(x_slopes) and id < len(d_slopes):
                sx = x_slopes[ix]
                sd = d_slopes[id]
                ratio = sd / sx if sx > 0 else 0
                
                print(f"{t:<12.1f} | {sx:<12.6f} | {sd:<12.6f} | {ratio:<10.2f}x")

if __name__ == "__main__":
    analyze_slope_granular_100k("results/marathon_100k_holy_grail.json", step_size=0.1)
