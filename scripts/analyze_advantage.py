import json
import numpy as np

def analyze_advantage_curve(json_path, loss_start=8.5, loss_end=4.5, step=0.05):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xavier = data['xavier']
    dpi = data['dpi']
    
    # Extraction des listes pour interpolation
    x_steps = [entry['step'] for entry in xavier]
    x_losses = [entry['loss'] for entry in xavier]
    
    d_steps = [entry['step'] for entry in dpi]
    d_losses = [entry['loss'] for entry in dpi]
    
    print(f"{'Target Loss':<12} | {'Step Xavier':<12} | {'Step DPI':<12} | {'Advantage (x)':<12} | {'Delta Steps':<12}")
    print("-" * 70)
    
    targets = np.arange(loss_start, loss_end, -step)
    
    results = []
    for target in targets:
        # Trouver le step correspondant par interpolation (plus précis que le step brut)
        # On utilise np.interp mais attention, les losses sont décroissantes, donc on inverse pour l'interpolation
        try:
            # Step où Xavier atteint Target
            s_x = np.interp(target, x_losses[::-1], x_steps[::-1])
            # Step où DPI atteint Target
            s_d = np.interp(target, d_losses[::-1], d_steps[::-1])
            
            # Si le step est hors limite (le modèle n'a pas encore atteint cette loss ou a commencé en dessous)
            if s_x == x_steps[0] or s_x == x_steps[-1] or s_d == d_steps[0] or s_d == d_steps[-1]:
                continue
                
            advantage = s_x / s_d
            delta = s_x - s_d
            
            results.append((target, s_x, s_d, advantage, delta))
            print(f"{target:<12.2f} | {s_x:<12.0f} | {s_d:<12.0f} | {advantage:<12.2f} | {delta:<12.0f}")
        except:
            continue

if __name__ == "__main__":
    analyze_advantage_curve("marathon_20k_results.json")
