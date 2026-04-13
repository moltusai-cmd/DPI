import json
import numpy as np

def analyze_slope_stable_100k(json_path, step_size=0.1, window=100):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    x_loss = np.array(data['xavier'])
    d_loss = np.array(data['dpi'])
    
    # Helper to get smoothed slope at a specific index
    def get_smoothed_slope(arr, idx, win):
        start = max(0, idx - win // 2)
        end = min(len(arr), idx + win // 2)
        if end - start < 2: return 0
        # Linear regression slope on the window
        y = arr[start:end]
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return -slope # Positive means loss is decreasing

    max_l = 9.7
    min_l = 4.0
    targets = np.arange(max_l, min_l, -0.5)
    
    print(f"\n{'Target Loss':<12} | {'Slope X (avg)':<15} | {'Slope D (avg)':<15} | {'Advantage'}")
    print("-" * 70)
    
    for t in targets:
        idx_x = np.where(x_loss <= t)[0]
        idx_d = np.where(d_loss <= t)[0]
        
        if len(idx_x) > 0 and len(idx_d) > 0:
            ix, id = idx_x[0], idx_d[0]
            sx = get_smoothed_slope(x_loss, ix, window)
            sd = get_smoothed_slope(d_loss, id, window)
            
            ratio = sd / sx if sx > 1e-7 else 0
            print(f"{t:<12.1f} | {sx:<15.7f} | {sd:<15.7f} | {ratio:<10.2f}x")

if __name__ == "__main__":
    analyze_slope_stable_100k("results/marathon_100k_holy_grail.json")
