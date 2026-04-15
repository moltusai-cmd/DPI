import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for direct imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import PID8Transformer
from initialize_dpi import initialize_dpi

def run_test():
    print("🚀 Starting DPI-14.1 Smoke Test...")
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # 2. Model Initialization (Small Scale)
    vocab_size = 1000
    d_model = 128
    n_layers = 4
    model = PID8Transformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=4, 
        d_mlp=512, 
        n_layers=n_layers
    ).to(device)
    
    print(f"  Model Instantiated: {n_layers} Layers, {d_model} d_model")
    
    # 3. Dummy Data Loader for Manifold Analysis
    # (Simulating [Batch, Seq_len])
    dummy_data = torch.randint(0, vocab_size, (16, 64))
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("  Dummy DataLoader Ready.")
    
    # 4. Running DPI-14.1
    print("\n--- DPI BOOTSTRAP START ---")
    try:
        initialize_dpi(
            model, 
            dataloader, 
            mode="v16.2"
        )
        print("--- DPI BOOTSTRAP COMPLETE ---")
        
        # 5. Verification: Check for NaNs
        nan_found = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ ERROR: NaNs detected in {name}!")
                nan_found = True
        
        if not nan_found:
            print("\n✅ SMOKE TEST PASSED: DPI-14.1 successfully initialized the model.")
        else:
            print("\n❌ SMOKE TEST FAILED: NaNs were found in the weights.")
            
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: An exception occurred during initialization.")
        print(f"Error Details: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
