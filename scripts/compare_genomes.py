import json
import numpy as np

def compare_genomes(dpi_path, ft_path):
    with open(dpi_path, "r") as f: dpi = json.load(f)
    with open(ft_path, "r") as f: ft = json.load(f)
    
    print(f"{'Layer':<6} | {'QK-Align (DPI)':<15} | {'QK-Align (FT)':<15} | {'Alpha W1 (DPI)':<15} | {'Alpha W1 (FT)':<15}")
    print("-" * 80)
    
    for i in range(len(dpi["layers"])):
        l_dpi = dpi["layers"][i]
        l_ft = ft["layers"][i]
        
        qk_dpi = l_dpi["components"]["qk_alignment"]
        qk_ft = l_ft["components"]["qk_alignment"]
        
        a_dpi = l_dpi["components"]["w1"]["spectral"]["alpha"]
        a_ft = l_ft["components"]["w1"]["spectral"]["alpha"]
        
        print(f"{i:<6} | {qk_dpi:15.4f} | {qk_ft:15.4f} | {a_dpi:15.4f} | {a_ft:15.4f}")

    print("\n--- Spectral Analysis (MLP Effective) ---")
    print(f"{'Layer':<6} | {'Eff Rank (DPI)':<15} | {'Eff Rank (FT)':<15} | {'Freq 1 (DPI)':<15} | {'Freq 1 (FT)':<15}")
    print("-" * 80)
    for i in range(len(dpi["layers"])):
        l_dpi = dpi["layers"][i]
        l_ft = ft["layers"][i]
        
        er_dpi = l_dpi["components"]["mlp_effective"]["spectral"]["eff_rank"]
        er_ft = l_ft["components"]["mlp_effective"]["spectral"]["eff_rank"]
        
        f_dpi = l_dpi["components"]["mlp_effective"]["harmonics"][0]["freq"]
        f_ft = l_ft["components"]["mlp_effective"]["harmonics"][0]["freq"]
        
        print(f"{i:<6} | {er_dpi:15.2f} | {er_ft:15.2f} | {f_dpi:15.4f} | {f_ft:15.4f}")

    print("\n--- Lexical Manifold (Embedding & Unembed Alpha) ---")
    print(f"Embedding Alpha | DPI: {dpi['embedding']['spectral']['alpha']:.4f} | FT: {ft['embedding']['spectral']['alpha']:.4f}")
    print(f"Unembed Alpha   | DPI: {dpi['unembed']['spectral']['alpha']:.4f} | FT: {ft['unembed']['spectral']['alpha']:.4f}")

if __name__ == "__main__":
    compare_genomes("GENOME_DPI.json", "GENOME_XAVIER.json")
