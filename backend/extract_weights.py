"""
Extract weights from full model file using manual unpickling
"""
import torch
import sys
from pathlib import Path

# Import model classes so PyTorch can find them during unpickling
sys.path.insert(0, str(Path(__file__).parent))
from chexca_model import (
    True_CHEXCA, ConvNeXtBaseBackbone, CTCAHybrid, 
    CHEXCA_Fusion, CHEXCA_GAT
)

# Make classes available in __main__ namespace for unpickling
import __main__
__main__.True_CHEXCA = True_CHEXCA
__main__.ConvNeXtBaseBackbone = ConvNeXtBaseBackbone
__main__.CTCAHybrid = CTCAHybrid
__main__.CHEXCA_Fusion = CHEXCA_Fusion
__main__.CHEXCA_GAT = CHEXCA_GAT

def extract_state_dict_manual():
    """Manually extract state dict from full model"""
    
    model_dir = Path(__file__).parent / "models"
    full_model_path = model_dir / "chexca_full_model.pth"
    state_dict_path = model_dir / "chexca_state_dict.pth"
    
    print(f"Loading checkpoint from: {full_model_path}")
    
    try:
        # Load with weights_only=False and without class reconstruction
        checkpoint = torch.load(
            full_model_path, 
            map_location='cpu',
            weights_only=False
        )
        
        print(f"Checkpoint type: {type(checkpoint)}")
        
        # Try to extract state dict
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            print("✓ Extracted state dict from model object")
        elif isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ Found state_dict key in checkpoint")
            else:
                state_dict = checkpoint
                print("✓ Checkpoint is already a state dict")
        else:
            print(f"✗ Unknown checkpoint format: {type(checkpoint)}")
            return False
        
        # Verify it's a valid state dict
        print(f"\nState dict keys (first 5): {list(state_dict.keys())[:5]}")
        print(f"Total parameters: {len(state_dict)}")
        
        # Save it
        torch.save(state_dict, state_dict_path)
        size_mb = state_dict_path.stat().st_size / (1024*1024)
        print(f"\n✓ Saved state dict to: {state_dict_path}")
        print(f"✓ File size: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CheXCA Weight Extractor")
    print("=" * 60)
    print()
    
    success = extract_state_dict_manual()
    
    print()
    if success:
        print("✓ SUCCESS!")
        print("\nUpdate config.py:")
        print("MODEL_PATH = BASE_DIR / 'models' / 'chexca_state_dict.pth'")
    else:
        print("✗ FAILED - See errors above")
