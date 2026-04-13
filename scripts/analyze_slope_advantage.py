import json
import numpy as np

def analyze_slope_advantage(json_path, loss_start=8.0, loss_end=4.5, step=0.05):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xavier = data['xavier']
    dpi = data['dpi']
    
    def get_slope_at_loss(history, target_loss):
        # On cherche le segment [i-1, i] qui contient la target_loss
        for i in range(1, len(history)):
            l_prev, l_curr = history[i-1]['loss'], history[i]['loss']
            # Comme la loss décroît, l_prev > l_curr
            if l_prev >= target_loss >= l_curr:
                # Pente locale : (L_curr - L_prev) / (Step_curr - Step_prev)
                # On veut une valeur positive pour la "vitesse de descente"
                slope = abs((l_curr - l_prev) / (history[i]['step'] - history[i-1]['step']))
                return slope
        return None

    print(f"{'Target Loss':<12} | {'Slope Xavier':<15} | {'Slope DPI':<15} | {'Slope Advantage (x)'}")
    print("-" * 70)
    
    targets = np.arange(loss_start, loss_end, -step)
    
    for target in targets:
        s_x = get_slope_at_loss(xavier, target)
        s_d = get_slope_at_loss(dpi, target)
        
        if s_x and s_d and s_x > 0:
            advantage = s_d / s_x
            print(f"{target:<12.2f} | {s_x:<15.6f} | {s_d:<15.6f} | {advantage:<15.2f}x")

if __name__ == "__main__":
    analyze_slope_advantage("marathon_20k_results.json")
