import json
import numpy as np

def analyze_multirun_stats(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xavier = data['xavier']
    dpi = data['dpi']
    
    steps = [1, 200, 400, 600, 800, 1000]
    
    print(f"{'Step':<6} | {'Xavier (Mean ± Std)':<25} | {'DPI (Mean ± Std)':<25} | {'Advantage'}")
    print("-" * 85)
    
    for s in steps:
        x_vals = [run[str(s)] for run in xavier]
        d_vals = [run[str(s)] for run in dpi]
        
        mx, sx = np.mean(x_vals), np.std(x_vals)
        md, sd = np.mean(d_vals), np.std(d_vals)
        
        # Formatage propre : Moyenne ± Ecart-Type
        x_str = f"{mx:.4f} ± {sx:.4f}"
        d_str = f"{md:.4f} ± {sd:.4f}"
        
        advantage = mx - md
        
        print(f"{s:<6} | {x_str:<25} | {d_str:<25} | {advantage:+.4f}")

if __name__ == "__main__":
    analyze_multirun_stats("code_multirun_results.json")
