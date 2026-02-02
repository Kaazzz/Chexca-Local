import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import logging
import sys
from pathlib import Path

# Add backend directory to path so we can import chexca_model
sys.path.insert(0, str(Path(__file__).parent))

import config
from chexca_model import load_chexca_model, safe_sigmoid, True_CHEXCA, build_convnext_backbone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChestXrayModel:
    """Wrapper for CheXCA model inference"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load trained model
        self.model = self._load_model()
        self.model.eval()
        
        # Disease classes (must match training order)
        self.disease_classes = config.DISEASE_CLASSES
        self.num_classes = len(self.disease_classes)
        
        # Image preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Model initialized for {self.num_classes} disease classes")
    
    def _load_model(self) -> nn.Module:
        """Load the trained CheXCA model"""
        try:
            logger.info(f"Loading model from: {config.MODEL_PATH}")
            
            # Method 1: Build architecture first, then load weights
            # This is the most reliable method for cross-environment loading
            try:
                logger.info("Building model architecture from scratch...")
                backbone = build_convnext_backbone()
                model = True_CHEXCA(backbone=backbone, num_classes=len(config.DISEASE_CLASSES))
                
                # Try to load as state dict first
                logger.info("Loading weights...")
                checkpoint = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=False)
                
                # Check if it's a full model or state dict
                if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
                    # It's a state dict
                    model.load_state_dict(checkpoint)
                    logger.info("✓ Loaded model from state dict")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # It's a checkpoint with state_dict key
                    model.load_state_dict(checkpoint['state_dict'])
                    logger.info("✓ Loaded model from checkpoint")
                elif hasattr(checkpoint, 'state_dict'):
                    # It's a full model object
                    model.load_state_dict(checkpoint.state_dict())
                    logger.info("✓ Loaded model from full model object")
                else:
                    raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
                
                return model
                
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                logger.error(f"\nTroubleshooting:")
                logger.error(f"1. Check that {config.MODEL_PATH} exists")
                logger.error(f"2. Run convert_model.py to extract state dict from full model")
                logger.error(f"3. Ensure model architecture matches training")
                raise RuntimeError(
                    f"Failed to load model from {config.MODEL_PATH}\n"
                    f"Error: {e}\n"
                    f"Run 'python convert_model.py' to convert your model file."
                )
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess image for model inference
        
        Args:
            image_path: Path to image file
        
        Returns:
            image_tensor: Preprocessed tensor (1, 3, 224, 224)
            original_image: Original PIL image
        """
        try:
            # Load and convert to RGB
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)
            
            return image_tensor.to(self.device), original_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, float]:
        """
        Run inference on a chest X-ray image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary mapping disease names to probabilities
        """
        image_tensor, _ = self.preprocess_image(image_path)
        
        with torch.no_grad():
            # Forward pass
            logits, metadata = self.model(image_tensor)
            
            # Apply sigmoid (trained with BCEWithLogitsLoss equivalent)
            probabilities = safe_sigmoid(logits).squeeze().cpu().numpy()
        
        # Handle single output case
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])
        
        # Create disease-probability mapping
        predictions = {
            disease: float(prob)
            for disease, prob in zip(self.disease_classes, probabilities)
        }
        
        return predictions
    
    def predict_with_attention(self, image_path: str) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Run inference and return attention maps
        
        Args:
            image_path: Path to image file
        
        Returns:
            predictions: Disease probabilities
            attention: CTCA attention weights (num_classes, H*W)
        """
        image_tensor, _ = self.preprocess_image(image_path)
        
        with torch.no_grad():
            logits, metadata = self.model(image_tensor)
            probabilities = safe_sigmoid(logits).squeeze().cpu().numpy()
            attention = metadata['ctca_attn'].squeeze()  # (num_classes, H*W)
        
        predictions = {
            disease: float(prob)
            for disease, prob in zip(self.disease_classes, probabilities)
        }
        
        return predictions, attention
    
    def get_top_predictions(self, predictions: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top K predictions sorted by probability
        
        Args:
            predictions: Disease probability dictionary
            top_k: Number of top predictions to return
        
        Returns:
            List of (disease, probability) tuples
        """
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_predictions[:top_k]
    
    def calculate_co_occurrence(self, predictions: Dict[str, float], threshold: float = 0.3) -> np.ndarray:
        """
        Calculate disease co-occurrence matrix based on predictions
        
        Note: For a single prediction, this creates a similarity-based matrix.
        In production with historical data, this would reflect actual co-occurrence patterns.
        
        Args:
            predictions: Disease probability dictionary
            threshold: Probability threshold for considering disease present
        
        Returns:
            Co-occurrence matrix (num_classes, num_classes)
        """
        diseases = list(predictions.keys())
        probs = np.array(list(predictions.values()))
        n = len(diseases)
        
        co_occurrence = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: probability of disease itself
                    co_occurrence[i][j] = probs[i]
                else:
                    # Off-diagonal: similarity measure
                    # High when both probabilities are similar (both high or both low)
                    similarity = 1 - abs(probs[i] - probs[j])
                    co_occurrence[i][j] = (probs[i] + probs[j]) * similarity / 2
        
        return co_occurrence


# ========================
# Global Model Instance
# ========================
_model_instance = None


def get_model() -> ChestXrayModel:
    """Get or create global model instance (singleton pattern)"""
    global _model_instance
    
    if _model_instance is None:
        logger.info("Initializing CheXCA model...")
        _model_instance = ChestXrayModel()
        logger.info("✓ Model ready for inference")
    
    return _model_instance
