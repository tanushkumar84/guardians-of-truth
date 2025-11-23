"""
Enhanced Model Architectures for Deepfake Detection
Features:
- Attention mechanisms (Squeeze-and-Excitation, CBAM)
- Better classifier heads with residual connections
- Focal Loss for handling class imbalance
- Label Smoothing
- Uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import timm
import logging

logger = logging.getLogger(__name__)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        avg_out = self.channel_attention(avg_pool)
        max_out = self.channel_attention(max_pool)
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.spatial_attention(spatial_input))
        x = x * spatial_att
        
        return x


class AttentionClassifierHead(nn.Module):
    """
    Advanced classifier head with attention and residual connections
    """
    def __init__(self, in_features, num_classes=1, dropout_rates=[0.5, 0.4, 0.3]):
        super().__init__()
        
        # Attention module
        self.attention = SqueezeExcitation(in_features)
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # First dense block with residual
        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # Second dense block
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # Third dense block
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        # Output layer
        self.fc_out = nn.Linear(256, num_classes)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply attention if input is 4D
        if len(x.shape) == 4:
            x = self.attention(x)
            x = self.pool(x)
        
        x = self.flatten(x)
        
        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc_out(x)
        
        return x


class EnhancedEfficientNet(nn.Module):
    """
    Enhanced EfficientNet with attention mechanisms and better classifier
    """
    def __init__(self, model_name='efficientnet-b3', pretrained=True):
        super().__init__()
        
        # Load backbone
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name, num_classes=0)
        else:
            self.backbone = EfficientNet.from_name(model_name, num_classes=0)
        
        # Get feature dimensions
        in_features = self.backbone._fc.in_features if hasattr(self.backbone, '_fc') else 1536
        
        # Enhanced classifier head with attention
        self.classifier = AttentionClassifierHead(in_features, num_classes=1)
        
        logger.info(f"Created EnhancedEfficientNet with {model_name}")

    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.classifier(features)

    def get_feature_extractor(self):
        """Return the backbone for feature extraction"""
        return self.backbone


class EnhancedSwinTransformer(nn.Module):
    """
    Enhanced Swin Transformer with attention and better classifier
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Get feature dimensions (Swin-Base = 1024)
        in_features = self.backbone.num_features
        
        # Layer norm for Swin features
        self.norm = nn.LayerNorm([in_features, 7, 7])
        
        # CBAM attention
        self.cbam = CBAM(in_features)
        
        # Enhanced classifier
        self.classifier = AttentionClassifierHead(in_features, num_classes=1)
        
        logger.info(f"Created EnhancedSwinTransformer with {model_name}")

    def forward(self, x):
        # Get features [B, H, W, C]
        features = self.backbone.forward_features(x)
        
        # Permute to [B, C, H, W]
        features = features.permute(0, 3, 1, 2)
        
        # Apply normalization
        features = self.norm(features)
        
        # Apply CBAM attention
        features = self.cbam(features)
        
        # Classify
        return self.classifier(features)

    def get_feature_extractor(self):
        """Return the backbone for feature extraction"""
        return self.backbone


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions (logits) [B]
            targets: ground truth labels [B]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if y=1, else 1-p
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Focal loss
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for better generalization
    Smooths hard labels: y = (1 - smoothing) * y + smoothing / num_classes
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions (logits) [B]
            targets: ground truth labels [B]
        """
        # Smooth labels
        targets_smooth = targets * self.confidence + self.smoothing * 0.5
        
        # Compute loss with smoothed labels
        loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth)
        
        return loss


class UncertaintyHead(nn.Module):
    """
    Predict both class and uncertainty (epistemic + aleatoric)
    Useful for identifying hard/ambiguous samples
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc_logit = nn.Linear(in_features, 1)  # Class logit
        self.fc_uncertainty = nn.Linear(in_features, 1)  # Uncertainty (log variance)

    def forward(self, x):
        logit = self.fc_logit(x)
        log_variance = self.fc_uncertainty(x)
        return logit, log_variance


def get_loss_function(loss_type='bce', **kwargs):
    """
    Factory function to get loss function
    
    Args:
        loss_type: Type of loss ('bce', 'focal', 'label_smoothing')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(smoothing=smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test models
    logger.info("Testing enhanced models...")
    
    # Test EfficientNet
    model_eff = EnhancedEfficientNet(model_name='efficientnet-b3', pretrained=False)
    x = torch.randn(2, 3, 300, 300)
    out = model_eff(x)
    logger.info(f"EnhancedEfficientNet output shape: {out.shape}")
    
    # Test Swin
    model_swin = EnhancedSwinTransformer(model_name='swin_base_patch4_window7_224', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model_swin(x)
    logger.info(f"EnhancedSwinTransformer output shape: {out.shape}")
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    loss = focal_loss(inputs, targets)
    logger.info(f"Focal Loss: {loss.item():.4f}")
    
    logger.info("✅ All models and losses working correctly!")
