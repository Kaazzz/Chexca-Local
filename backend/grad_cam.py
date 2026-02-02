import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
import logging

import config

logger = logging.getLogger(__name__)


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for CheXCA model
    Works with ConvNeXt backbone + CTCA architecture
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layer"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map using Grad-CAM++
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            class_idx: Target class index (if None, uses highest probability)
        
        Returns:
            CAM as numpy array (H, W), values in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        logits, _ = self.model(image_tensor)
        
        if class_idx is None:
            # Use class with highest probability
            class_idx = torch.sigmoid(logits).argmax(dim=1).item()
        
        # Ensure gradients are enabled
        self.model.zero_grad()
        
        # Get class score
        class_score = logits[0, class_idx]
        
        # Backward pass
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Grad-CAM++ weighting
        # Calculate alpha (importance weights for each channel)
        numerator = gradients ** 2
        denominator = 2 * (gradients ** 2) + (activations * (gradients ** 3)).sum(axis=(1, 2), keepdims=True)
        denominator = np.where(denominator != 0, denominator, 1e-10)
        alpha = numerator / denominator
        
        # Weight gradients by alpha
        weights = (alpha * np.maximum(gradients, 0)).sum(axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Resize to input size
        cam = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_heatmap(self, image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original PIL image
            cam: Class activation map (H, W)
            alpha: Overlay transparency (0=original, 1=heatmap only)
        
        Returns:
            PIL Image with heatmap overlay
        """
        # Resize image if needed
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        image_np = np.array(image)
        
        # Create colored heatmap (JET colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlay)


def generate_gradcam(
    model,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class: Optional[int] = None
) -> Tuple[np.ndarray, Image.Image]:
    """
    Generate Grad-CAM++ visualization for CheXCA model
    
    Args:
        model: Trained CheXCA model
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
        original_image: Original PIL image
        target_class: Target disease class index (if None, uses top prediction)
    
    Returns:
        cam: Class activation map (224, 224), values in [0, 1]
        overlay_image: Original image with heatmap overlay
    """
    try:
        # Get target layer (last conv layer in ConvNeXt backbone)
        # For ConvNeXt, it's typically the last layer before global pooling
        target_layer = None
        
        # Try to find the appropriate layer
        if hasattr(model, 'backbone'):
            # Navigate ConvNeXt structure
            if hasattr(model.backbone, 'model'):
                if hasattr(model.backbone.model, 'stages'):
                    # Last stage of ConvNeXt
                    target_layer = model.backbone.model.stages[-1]
                    logger.info("Using ConvNeXt last stage for Grad-CAM")
        
        if target_layer is None:
            logger.warning("Could not find optimal target layer, using fallback")
            # Fallback: use last convolutional layer we can find
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
                    target_layer = module
        
        if target_layer is None:
            raise ValueError("Could not find appropriate target layer for Grad-CAM")
        
        # Create Grad-CAM instance
        grad_cam = GradCAMPlusPlus(model, target_layer)
        
        # Generate CAM
        cam = grad_cam.generate_cam(image_tensor, target_class)
        
        # Create overlay
        overlay_image = grad_cam.overlay_heatmap(original_image, cam, alpha=0.4)
        
        return cam, overlay_image
    
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        logger.info("Returning blank heatmap as fallback")
        
        # Return blank/neutral heatmap as fallback
        cam = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        overlay_image = original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        return cam, overlay_image
