"""
XceptionNet Model Architecture for Deepfake Detection
Train from scratch as third ensemble component
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Xception Block with Residual Connection"""
    def __init__(self, in_filters, out_filters, strides=1, grow_first=True, activate_first=True):
        super(XceptionBlock, self).__init__()
        
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False),
                nn.BatchNorm2d(out_filters)
            )
        else:
            self.skip = None
        
        self.relu = nn.ReLU(inplace=False)
        
        rep = []
        filters = in_filters
        
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        
        for _ in range(2):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, 1))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1))
            rep.append(nn.BatchNorm2d(out_filters))
        
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        
        self.rep = nn.Sequential(*rep)
        
    def forward(self, x):
        output = self.rep(x)
        
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
        
        output += skip
        return output


class XceptionNet(nn.Module):
    """
    XceptionNet for Deepfake Detection
    Input: 299x299x3
    Output: 2 classes (Real/Fake)
    """
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(XceptionNet, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        
        self.block1 = XceptionBlock(64, 128, strides=2, grow_first=True, activate_first=True)
        self.block2 = XceptionBlock(128, 256, strides=2, grow_first=True, activate_first=True)
        self.block3 = XceptionBlock(256, 728, strides=2, grow_first=True, activate_first=True)
        
        # Middle flow (8 blocks)
        self.middle_flow = nn.Sequential(
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
            XceptionBlock(728, 728, strides=1, grow_first=True, activate_first=True),
        )
        
        # Exit flow
        self.block4 = XceptionBlock(728, 1024, strides=2, grow_first=False, activate_first=True)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=False)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=False)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.middle_flow(x)
        
        # Exit flow
        x = self.block4(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps for visualization"""
        features = {}
        
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        features['entry_1'] = x
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        features['entry_2'] = x
        
        x = self.block1(x)
        features['block_1'] = x
        x = self.block2(x)
        features['block_2'] = x
        x = self.block3(x)
        features['block_3'] = x
        
        # Middle flow
        x = self.middle_flow(x)
        features['middle_flow'] = x
        
        # Exit flow
        x = self.block4(x)
        features['exit'] = x
        
        return features


def create_xception(num_classes=2, dropout_rate=0.5, pretrained=False):
    """
    Create XceptionNet model
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        dropout_rate: Dropout rate before final FC layer (default: 0.5)
        pretrained: Not used, always train from scratch (default: False)
    
    Returns:
        XceptionNet model
    """
    model = XceptionNet(num_classes=num_classes, dropout_rate=dropout_rate)
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing XceptionNet Architecture...")
    
    model = create_xception(num_classes=2, dropout_rate=0.5)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 299, 299)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_feature_maps(dummy_input)
    print("\nFeature map shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print("\n✅ XceptionNet architecture test passed!")
