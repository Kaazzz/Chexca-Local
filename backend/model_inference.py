import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import config
from chexca_model import load_chexca_model


class ChestXrayModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and get actual number of classes
        self.model, self.num_classes = self._load_model()
        self.model.eval()

        # Update disease classes if needed
        if self.num_classes != len(config.DISEASE_CLASSES):
            print(f"WARNING: Model outputs {self.num_classes} classes but config has {len(config.DISEASE_CLASSES)}")
            if self.num_classes == 1:
                print("Model appears to be binary classifier")
                self.disease_classes = ["Disease_Present"]
            else:
                self.disease_classes = config.DISEASE_CLASSES[:self.num_classes]
        else:
            self.disease_classes = config.DISEASE_CLASSES

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Load the trained PyTorch model"""
        print("Loading CheXCA custom model...")
        model, num_classes = load_chexca_model(
            config.MODEL_PATH,
            num_classes=len(config.DISEASE_CLASSES),
            device=self.device
        )
        print(f"Model loaded successfully with {num_classes} output classes")
        return model, num_classes

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Preprocess image for model inference"""
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()

        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device), original_image

    def predict(self, image_path: str) -> Dict[str, float]:
        """Run inference on an image"""
        image_tensor, _ = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Handle single output
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])

        # Create disease-probability mapping
        predictions = {
            disease: float(prob)
            for disease, prob in zip(self.disease_classes, probabilities)
        }

        return predictions

    def get_top_predictions(self, predictions: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K predictions"""
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]

    def calculate_co_occurrence(self, predictions: Dict[str, float], threshold: float = 0.5) -> np.ndarray:
        """
        Calculate disease co-occurrence matrix
        For a single prediction, we create a simplified co-occurrence based on probability similarities
        In production, this would be calculated from a dataset of multiple patients
        """
        diseases = list(predictions.keys())
        probs = np.array(list(predictions.values()))

        # Create a co-occurrence matrix based on probability similarities
        # Higher values = diseases that tend to occur together
        co_occurrence = np.zeros((len(diseases), len(diseases)))

        for i in range(len(diseases)):
            for j in range(len(diseases)):
                if i == j:
                    co_occurrence[i][j] = probs[i]
                else:
                    # Similarity measure: both high or both low probabilities
                    similarity = 1 - abs(probs[i] - probs[j])
                    co_occurrence[i][j] = (probs[i] + probs[j]) * similarity / 2

        return co_occurrence


# Global model instance
model_instance = None

def get_model() -> ChestXrayModel:
    """Get or create model instance"""
    global model_instance
    if model_instance is None:
        model_instance = ChestXrayModel()
    return model_instance
