import streamlit as st
import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path
from .utils_eff import DeepfakeEfficientNet
from .utils_swin import DeepfakeSwin

# Add training directory to path for model imports
sys.path.append(str(Path(__file__).parent.parent / 'training'))
from xception_model import create_xception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightDeepfakeCNN(nn.Module):
    """Lightweight CNN for deepfake detection - 4 block architecture"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@st.cache_resource
def get_cached_model(model_path, model_type):
    """Cache and share model instances across sessions"""
    try:
        if model_type == "efficientnet":
            model = DeepfakeEfficientNet()
        elif model_type == "swin":
            model = DeepfakeSwin()
        elif model_type == "xception":
            model = create_xception(num_classes=2, dropout_rate=0.5)
        elif model_type == "mobilenet":
            import torchvision.models as models
            model = models.mobilenet_v2(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.last_channel, 2)
            )
        elif model_type == "custom":
            model = LightweightDeepfakeCNN(num_classes=2)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
            
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None