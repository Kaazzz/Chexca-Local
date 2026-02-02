import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "chexca_state_dict.pth"  # Extracted state dict
UPLOAD_DIR = BASE_DIR / "uploads"

# Create upload directory if it doesn't exist
UPLOAD_DIR.mkdir(exist_ok=True)

# NIH ChestX-ray14 disease classes
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

# Model configuration
IMAGE_SIZE = 224
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# Mock mode DISABLED - using real trained model
MOCK_MODE = False
