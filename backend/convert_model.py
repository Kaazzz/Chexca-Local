"""
Script to convert full model to state dict for proper loading
Run this once to extract weights from chexca_full_model.pth
"""
import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chexca_model import True_CHEXCA, build_convnext_backbone

def convert_full_model_to_state_dict():
    """Extract state dict from full model file"""
    
    model_dir = Path(__file__).parent / "models"
    full_model_path = model_dir / "chexca_full_model.pth"
    state_dict_path = model_dir / "chexca_state_dict.pth"
    
    print(f"Loading full model from: {full_model_path}")
    
    try:
        # Try to load the full model with weights_only=False
        # This requires the model class to be importable
        model = torch.load(full_model_path, map_location='cpu', weights_only=False)
        print("✓ Successfully loaded full model")
        
        # Extract state dict
        state_dict = model.state_dict()
        
        # Save state dict
        torch.save(state_dict, state_dict_path)
        print(f"✓ Saved state dict to: {state_dict_path}")
        print(f"✓ File size: {state_dict_path.stat().st_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert model: {e}")
        print("\nTrying alternative method...")
        
        try:
            # Alternative: manually register the class and try again
            import torch.serialization
            torch.serialization.add_safe_globals([True_CHEXCA])
            
            model = torch.load(full_model_path, map_location='cpu', weights_only=False)
            state_dict = model.state_dict()
            torch.save(state_dict, state_dict_path)
            print(f"✓ Saved state dict to: {state_dict_path}")
            return True
            
        except Exception as e2:
            print(f"✗ Alternative method also failed: {e2}")
            print("\nThe full model file may have been saved with incompatible settings.")
            print("Please re-save your model using one of these methods:")
            print("1. Save state dict only: torch.save(model.state_dict(), 'path.pth')")
            print("2. Save with proper imports available")
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("CheXCA Model Converter")
    print("=" * 60)
    print()
    
    success = convert_full_model_to_state_dict()
    
    print()
    if success:
        print("✓ Conversion successful!")
        print("\nNext steps:")
        print("1. Update config.py to use: MODEL_PATH = BASE_DIR / 'models' / 'chexca_state_dict.pth'")
        print("2. Restart your backend server")
    else:
        print("✗ Conversion failed. See error messages above.")
    print()
