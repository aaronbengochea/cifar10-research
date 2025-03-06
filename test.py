import torch
import os

checkpoint_file = 'checkpoint/ckpt.pth'
if os.path.isfile(checkpoint_file):
    # Load the checkpoint (using map_location='cpu' to ensure it loads even if CUDA isn't available)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Print the keys and a summary of the checkpoint content
    print("Checkpoint keys:", list(checkpoint.keys()))
    print("\nFull checkpoint contents:")
    for key, value in checkpoint.items():
        if key == 'net':
            print(f"{key}: state_dict with {len(value)} keys")
        else:
            print(f"{key}: {value}")
else:
    print("Checkpoint file not found.")