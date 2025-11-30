import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple
import config


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate Class Activation Map"""
        self.model.eval()

        # Forward pass
        output = self.model(image_tensor)

        if class_idx is None:
            # Use the class with highest probability
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]

        # Weight activations by gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_heatmap(self, image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """Overlay heatmap on original image"""
        # Resize image if needed
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        image_np = np.array(image)

        # Create colored heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

        return Image.fromarray(overlay)


def generate_gradcam(model, image_tensor: torch.Tensor, original_image: Image.Image,
                     target_class: int = None) -> Tuple[np.ndarray, Image.Image]:
    """
    Generate Grad-CAM visualization

    Args:
        model: The trained model
        image_tensor: Preprocessed image tensor
        original_image: Original PIL image
        target_class: Target class index (if None, uses highest probability class)

    Returns:
        cam: Class activation map
        overlay_image: Original image with heatmap overlay
    """
    # Get the last convolutional layer
    # For DenseNet121, it's features.denseblock4
    try:
        target_layer = model.features.denseblock4
    except:
        # Fallback for different architectures
        target_layer = list(model.children())[-2]

    # Create Grad-CAM instance
    grad_cam = GradCAM(model, target_layer)

    # Generate CAM
    cam = grad_cam.generate_cam(image_tensor, target_class)

    # Create overlay
    overlay_image = grad_cam.overlay_heatmap(original_image, cam)

    return cam, overlay_image
