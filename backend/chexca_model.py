"""
CheXCA Model Architecture
ConvNeXt-Base + CTCA (Class Token Cross-Attention) + GAT Fusion
Trained on NIH ChestX-ray14 dataset (14 diseases)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
from typing import Tuple, Dict

# Attempt to import PyTorch Geometric for GAT
USE_PYG = True
try:
    from torch_geometric.nn import GATv2Conv
except Exception as e:
    USE_PYG = False
    warnings.warn(
        "torch_geometric not available. Falling back to MLP-based GAT. "
        "Install PyG for full GAT behaviour: pip install torch-geometric\n" + str(e)
    )


# ========================
# Utility Functions
# ========================
def safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Stable sigmoid used during training"""
    return torch.clamp(torch.sigmoid(x), 1e-6, 1 - 1e-6)


# ========================
# ConvNeXt Backbone
# ========================
class ConvNeXtBaseBackbone(nn.Module):
    """ConvNeXt-Base backbone for feature extraction"""
    
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "convnext_base.fb_in22k",  # ImageNet-22K pretrained
            pretrained=True,
            in_chans=3,
            num_classes=0,  # Remove classifier head
            global_pool=""  # Keep spatial features
        )
    
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image"""
        return self.model.forward_features(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ========================
# CTCA (Class Token Cross-Attention) - Hybrid
# ========================
class CTCAHybrid(nn.Module):
    """
    Hybrid Class Token Cross-Attention module
    Combines class-to-patch and class-to-class attention
    """
    
    def __init__(self, in_channels: int, num_classes: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Learnable class tokens (one per disease)
        self.class_tokens = nn.Parameter(torch.randn(num_classes, in_channels) * 0.02)
        
        # Multi-head attention layers
        self.mha_cls_patch = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.mha_cls_cls = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Normalization and MLP
        self.ln_tokens = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map (B, C, H, W)
        
        Returns:
            tokens: Updated class tokens (B, num_classes, C)
            attn_cp: Class-to-patch attention weights (B, num_classes, H*W)
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions to patches
        patches = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Expand class tokens for batch
        cls = self.class_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_classes, C)
        
        # Class-to-patch attention
        attn_cp_out, attn_cp = self.mha_cls_patch(
            query=cls,
            key=patches,
            value=patches,
            need_weights=True
        )
        
        # Class-to-class attention
        attn_cc_out, _ = self.mha_cls_cls(
            query=cls,
            key=cls,
            value=cls,
            need_weights=False
        )
        
        # Combine and refine tokens
        tokens = cls + attn_cp_out + attn_cc_out
        tokens = tokens + self.mlp(self.ln_tokens(tokens))
        
        return tokens, attn_cp


# ========================
# GAT (Graph Attention Network) Fusion
# ========================
if USE_PYG:
    class CHEXCA_GAT(nn.Module):
        """Graph Attention Network for disease relationship modeling"""
        
        def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, heads: int = 2, num_classes: int = 14):
            super().__init__()
            self.num_classes = num_classes
            
            # GAT layers
            self.gat1 = GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=heads,
                dropout=0.1
            )
            self.gat2 = GATv2Conv(
                in_channels=hidden_dim * heads,
                out_channels=in_dim,
                heads=1,
                dropout=0.1
            )
            self.act = nn.ReLU()
            
            # Create fully connected disease graph
            with torch.no_grad():
                base_edges = torch.combinations(torch.arange(self.num_classes), r=2).T
                base_edges = torch.cat([base_edges, base_edges.flip(0)], dim=1)
            self.register_buffer("base_edges", base_edges, persistent=False)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Token features (B, N, C)
            
            Returns:
                Updated token features (B, N, C)
            """
            B, N, C = x.shape
            device = x.device
            
            # Create batch graph edges
            offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
            batch_edges = self.base_edges.unsqueeze(0) + offsets
            edge_index = batch_edges.permute(1, 0, 2).reshape(2, -1)
            
            # Flatten batch for PyG
            x_flat = x.reshape(B * N, C)
            
            # Apply GAT layers
            x_flat = self.gat1(x_flat, edge_index)
            x_flat = self.act(x_flat)
            x_flat = self.gat2(x_flat, edge_index)
            
            return x_flat.reshape(B, N, C)

else:
    class CHEXCA_GAT(nn.Module):
        """Fallback GAT using MLP when PyG not available"""
        
        def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, heads: int = 2, num_classes: int = 14):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp(x)


class CHEXCA_Fusion(nn.Module):
    """Fusion module combining GAT and MLP with residual connections"""
    
    def __init__(self, dim: int = 1024, num_classes: int = 14):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gat = CHEXCA_GAT(in_dim=dim, num_classes=num_classes)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply GAT and MLP with residual connections"""
        tokens = tokens + self.gat(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


# ========================
# Complete CheXCA Model
# ========================
class True_CHEXCA(nn.Module):
    """
    Complete CheXCA Architecture
    ConvNeXt-Base + CTCA + GAT Fusion + Per-class Classifier
    """
    
    def __init__(self, backbone: ConvNeXtBaseBackbone, num_classes: int = 14):
        super().__init__()
        self.backbone = backbone
        self.ctca = CTCAHybrid(in_channels=1024, num_classes=num_classes)
        self.fusion = CHEXCA_Fusion(dim=1024, num_classes=num_classes)
        
        # Per-class classifier (outputs single logit per disease)
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input image (B, 3, 224, 224)
        
        Returns:
            logits: Disease logits (B, num_classes)
            metadata: Dictionary containing intermediate outputs
        """
        # Extract features
        feats = self.backbone.features(x)  # (B, 1024, 7, 7)
        
        # Class token attention
        tokens, attn_cp = self.ctca(feats)  # (B, num_classes, 1024)
        
        # GAT fusion
        tokens = self.fusion(tokens)  # (B, num_classes, 1024)
        
        # Classify each disease
        logits = self.classifier(tokens).squeeze(-1)  # (B, num_classes)
        
        return logits, {"ctca_attn": attn_cp, "tokens": tokens}


# ========================
# Model Loading Functions
# ========================
def build_convnext_backbone() -> ConvNeXtBaseBackbone:
    """Build and initialize ConvNeXt backbone"""
    print("Loading ConvNeXt-Base backbone (ImageNet-22K pretrained)...")
    backbone = ConvNeXtBaseBackbone()
    return backbone


def load_chexca_model(checkpoint_path: str, num_classes: int = 14, device: str = 'cpu') -> True_CHEXCA:
    """
    Load complete CheXCA model from checkpoint
    
    Args:
        checkpoint_path: Path to saved model (.pth file)
        num_classes: Number of disease classes
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading CheXCA model from: {checkpoint_path}")
    
    try:
        # Try loading full model first (recommended)
        # Set weights_only=False for PyTorch 2.6+ compatibility with full models
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("✓ Loaded full model (architecture + weights)")
        
    except Exception as e:
        print(f"Full model loading failed, trying state dict...")
        print(f"Error: {e}")
        
        # Fallback: load state dict
        try:
            backbone = build_convnext_backbone()
            model = True_CHEXCA(backbone=backbone, num_classes=num_classes)
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print("✓ Loaded model from state dict")
            
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model. Tried both full model and state dict loading.\n"
                f"Full model error: {e}\n"
                f"State dict error: {e2}"
            )
    
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
    print(f"✓ Model expects {num_classes} disease classes")
    
    return model


# ========================
# For backwards compatibility
# ========================
class CheXCAModel(True_CHEXCA):
    """Alias for backwards compatibility"""
    pass
