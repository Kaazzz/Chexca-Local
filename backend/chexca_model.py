"""
Custom CheXCA model architecture
"""
import torch
import torch.nn as nn
from timm import create_model


class CheXCAModel(nn.Module):
    """
    Custom CheXCA architecture with ConvNeXt backbone
    """
    def __init__(self, num_classes=14):
        super().__init__()

        # ConvNeXt backbone (using timm)
        self.backbone = create_model(
            'convnext_base',
            pretrained=False,
            num_classes=0,  # Remove head, we'll use custom classifier
            global_pool=''
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024),
            nn.GELU()
        )

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)

        # Global average pooling
        features = features.mean(dim=[-2, -1])  # B x C x H x W -> B x C

        # Fusion
        features = self.fusion(features)

        # Classifier
        output = self.classifier(features)

        return output


def load_chexca_model(checkpoint_path, num_classes=14, device='cpu'):
    """
    Load CheXCA model from checkpoint

    Args:
        checkpoint_path: Path to .pth file
        num_classes: Number of output classes (default: 14 for NIH)
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if we need to adjust num_classes based on checkpoint
    if 'classifier.3.weight' in checkpoint:
        actual_classes = checkpoint['classifier.3.weight'].shape[0]
        print(f"Model was trained with {actual_classes} output class(es)")

        if actual_classes != num_classes:
            print(f"WARNING: Requested {num_classes} classes but model has {actual_classes}")
            print(f"Using {actual_classes} classes from checkpoint")
            num_classes = actual_classes

    # Create model
    model = CheXCAModel(num_classes=num_classes)

    # Load state dict
    model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    return model, num_classes
