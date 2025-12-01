import torch
import sys
from pathlib import Path
from chexca_model import load_chexca_model
import config

# Test the new model
model_path = Path(__file__).parent / "models" / "chexca_new.pth"
print(f"Loading model: {model_path}")

try:
    model, num_classes = load_chexca_model(model_path, num_classes=14, device='cpu')
    print(f"[OK] Model loaded successfully")
    print(f"  Number of classes detected: {num_classes}")

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"\nTesting with dummy input shape: {dummy_input.shape}")

    # Run inference
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n{'='*60}")
    print(f"OUTPUT ANALYSIS")
    print(f"{'='*60}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: torch.Size([1, 14]) for 14 classes")

    if output.shape[-1] == 14:
        print("\n[SUCCESS] Model outputs 14 classes!")
    elif output.shape[-1] == 1:
        print("\n[WARNING] Model outputs only 1 class")
        print("  This is a binary classifier, not multi-label (14 classes)")
    else:
        print(f"\n[INFO] Model outputs {output.shape[-1]} classes")

    # Apply sigmoid to see probabilities
    probs = torch.sigmoid(output)
    print(f"\nSample probabilities: {probs[0].tolist()}")

except Exception as e:
    print(f"\n[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()
