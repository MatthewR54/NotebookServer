import torch
from pathlib import Path

checkpoint_path = Path("models/mirage_model_6_d3/model_epoch_best.pt")

try:
    state = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(state, dict):
        print("Checkpoint is a dictionary")
        print(f"Keys: {list(state.keys())}")
        print(f"Total keys: {len(state)}")
        
        # Show shapes of tensors
        for key in list(state.keys())[:5]:  # Print first 5
            value = state[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"Checkpoint is not a dict, type: {type(state)}")
        if isinstance(state, torch.Tensor):
            print(f"  Shape: {state.shape}")
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
