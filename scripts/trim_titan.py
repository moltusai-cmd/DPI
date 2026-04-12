import torch
import os

def trim_and_bf16_streaming(input_path, output_path):
    print(f"Loading {input_path} using mmap (Memory Mapping)...")
    # mmap=True is the key to load 60GB on 64GB RAM!
    state_dict = torch.load(input_path, map_location='cpu', weights_only=True, mmap=True)
    
    clean_dict = {}
    print(f"Streaming layers, cleaning and converting to BF16...")
    for k, v in state_dict.items():
        # .clone() to break the reference to the big parent tensor
        # .to(torch.bfloat16) to reduce size
        clean_dict[k] = v.clone().detach().to(torch.bfloat16)
        
    print(f"Saving cleaned model to {output_path}...")
    # This might still take some RAM during serialization, but much less!
    torch.save(clean_dict, output_path)
    
    # Check size
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Cleaned Model Saved: {output_path}")
    print(f"Final Size: {size_gb:.2f} GB (Target: 16.38 GB)")

if __name__ == "__main__":
    if os.path.exists("dpi_8b_init.pt"):
        trim_and_bf16_streaming("dpi_8b_init.pt", "dpi_8b_bf16.pt")
    else:
        print("Error: dpi_8b_init.pt not found.")
