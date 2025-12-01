import torch
import sys
from pathlib import Path

# Allow passing model path as argument
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = Path(__file__).parent / "models" / "chexca_new.pth"

print(f"Loading model file: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')

print("\n" + "="*60)
print("MODEL FILE INSPECTION")
print("="*60)

if isinstance(checkpoint, dict):
    print("\n[OK] File contains a state_dict (weights only)")
    print(f"\nNumber of parameters: {len(checkpoint)}")

    print("\n--- First 10 parameter names ---")
    for i, key in enumerate(list(checkpoint.keys())[:10]):
        shape = checkpoint[key].shape
        print(f"{i+1}. {key:50s} -> {shape}")

    print("\n--- Last 10 parameter names ---")
    for i, key in enumerate(list(checkpoint.keys())[-10:]):
        shape = checkpoint[key].shape
        print(f"{i+1}. {key:50s} -> {shape}")

    # Try to detect architecture from key names
    keys = list(checkpoint.keys())
    first_key = keys[0]

    print("\n" + "="*60)
    print("ARCHITECTURE DETECTION")
    print("="*60)

    if 'features.conv0' in keys or 'features.denseblock1' in keys:
        print("[OK] Likely DenseNet architecture")
        if 'features.denseblock4' in keys:
            print("  - Probably DenseNet121, 169, or 201")
    elif 'layer1' in first_key or 'layer2' in first_key:
        print("[OK] Likely ResNet architecture")
    elif 'blocks' in first_key:
        print("[OK] Likely EfficientNet architecture")
    elif 'stages' in first_key:
        print("[OK] Likely RegNet or other staged architecture")
    else:
        print("[?] Unknown architecture")

    # Check for classifier/fc layer
    print("\n--- Output layer detection ---")
    for key in keys:
        if 'classifier' in key or 'fc' in key or 'head' in key:
            print(f"Output layer: {key} -> {checkpoint[key].shape}")

else:
    print("\n[OK] File contains a complete model")
    print(f"Model type: {type(checkpoint)}")
    print(f"Model class: {checkpoint.__class__.__name__}")

print("\n" + "="*60)
