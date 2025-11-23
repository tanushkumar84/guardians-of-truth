"""
Improved Model Inference with Better Fake Detection
Addresses false negatives by:
1. Lowering detection threshold
2. Better ensemble voting
3. Enhanced preprocessing
4. Confidence recalibration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedPredictor:
    """Enhanced predictor with better fake detection sensitivity"""
    
    def __init__(self, 
                 fake_threshold: float = 0.45,  # Lower threshold = more sensitive to fakes
                 ensemble_mode: str = 'aggressive',  # 'aggressive' or 'conservative'
                 confidence_boost: float = 1.1):  # Boost fake confidence
        """
        Args:
            fake_threshold: Threshold for fake detection (lower = more sensitive)
            ensemble_mode: 
                - 'aggressive': If ANY model says fake, classify as fake
                - 'conservative': Both models must agree
            confidence_boost: Multiply fake predictions by this factor
        """
        self.fake_threshold = fake_threshold
        self.ensemble_mode = ensemble_mode
        self.confidence_boost = confidence_boost
        
    def apply_confidence_recalibration(self, probability: float, is_fake: bool) -> float:
        """
        Recalibrate confidence scores to be more sensitive to fakes
        
        Args:
            probability: Raw model probability
            is_fake: Whether prediction is fake
            
        Returns:
            Recalibrated probability
        """
        if is_fake:
            # Boost fake confidence
            probability = min(1.0, probability * self.confidence_boost)
        else:
            # Slightly reduce real confidence to be more cautious
            probability = probability * 0.95
            
        return probability
    
    def enhanced_single_prediction(self, model: nn.Module, image_tensor: torch.Tensor, 
                                   device: torch.device) -> Tuple[str, float]:
        """
        Make prediction with improved sensitivity
        
        Returns:
            (prediction, probability)
        """
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            
            if isinstance(output, tuple):
                output = output[0]
            
            # Handle different output shapes
            # If output has 2 elements [real_prob, fake_prob], take the fake probability
            if output.numel() == 2:
                # Output is [batch, 2] or [2] - get fake probability (index 1)
                probability = torch.softmax(output, dim=-1)[..., 1].item()
            else:
                # Output is [batch, 1] or [1] - apply sigmoid
                probability = torch.sigmoid(output).squeeze().item()
            
            # Use adjusted threshold
            is_fake = probability > self.fake_threshold
            prediction = "FAKE" if is_fake else "REAL"
            
            # Apply recalibration
            probability = self.apply_confidence_recalibration(probability, is_fake)
            
            return prediction, probability
    
    def ensemble_predict(self, predictions: list) -> Tuple[str, float, dict]:
        """
        Improved ensemble prediction with better fake detection
        
        Args:
            predictions: List of (model_name, prediction, probability) tuples
            
        Returns:
            (final_prediction, confidence, details)
        """
        fake_count = sum(1 for _, pred, _ in predictions if pred == "FAKE")
        total_models = len(predictions)
        
        # Extract probabilities
        probs = [prob for _, _, prob in predictions]
        avg_prob = np.mean(probs)
        max_prob = np.max(probs)
        
        # Aggressive mode: If ANY model detects fake with high confidence
        if self.ensemble_mode == 'aggressive':
            # Check if any model strongly believes it's fake
            strong_fake_detections = [
                prob for _, pred, prob in predictions 
                if pred == "FAKE" and prob > 0.55
            ]
            
            if strong_fake_detections:
                final_prediction = "FAKE"
                confidence = max(strong_fake_detections)
            elif fake_count > 0:
                # At least one model says fake
                final_prediction = "FAKE"
                confidence = max_prob
            else:
                final_prediction = "REAL"
                confidence = 1.0 - max_prob
        
        # Conservative mode: Require majority
        else:
            if fake_count >= (total_models / 2):
                final_prediction = "FAKE"
                confidence = avg_prob
            else:
                final_prediction = "REAL"
                confidence = 1.0 - avg_prob
        
        details = {
            'fake_count': fake_count,
            'total_models': total_models,
            'avg_probability': avg_prob,
            'max_probability': max_prob,
            'individual_predictions': predictions
        }
        
        return final_prediction, confidence, details
    
    def aggregate_video_predictions(self, frame_predictions: list, 
                                   min_fake_ratio: float = 0.3) -> Tuple[str, float, dict]:
        """
        Aggregate predictions across video frames
        More sensitive to fake frames
        
        Args:
            frame_predictions: List of frame predictions
            min_fake_ratio: Minimum ratio of fake frames to classify video as fake
            
        Returns:
            (prediction, confidence, details)
        """
        if not frame_predictions:
            return "UNKNOWN", 0.0, {}
        
        fake_frames = sum(1 for pred in frame_predictions if pred == "FAKE")
        total_frames = len(frame_predictions)
        fake_ratio = fake_frames / total_frames
        
        # If significant portion of frames are fake, classify entire video as fake
        if fake_ratio >= min_fake_ratio:
            prediction = "FAKE"
            # Confidence based on how many frames are fake
            confidence = min(0.95, 0.6 + (fake_ratio * 0.35))
        else:
            prediction = "REAL"
            confidence = 1.0 - fake_ratio
        
        details = {
            'fake_frames': fake_frames,
            'total_frames': total_frames,
            'fake_ratio': fake_ratio,
            'threshold_used': min_fake_ratio
        }
        
        return prediction, confidence, details


def create_improved_predictor(sensitivity: str = 'high') -> ImprovedPredictor:
    """
    Factory function to create predictor with different sensitivity levels
    
    Args:
        sensitivity: 'low', 'medium', 'high', 'very_high'
    """
    configs = {
        'low': {
            'fake_threshold': 0.5,
            'ensemble_mode': 'conservative',
            'confidence_boost': 1.0
        },
        'medium': {
            'fake_threshold': 0.48,
            'ensemble_mode': 'conservative',
            'confidence_boost': 1.05
        },
        'high': {
            'fake_threshold': 0.45,
            'ensemble_mode': 'aggressive',
            'confidence_boost': 1.1
        },
        'very_high': {
            'fake_threshold': 0.40,
            'ensemble_mode': 'aggressive',
            'confidence_boost': 1.15
        }
    }
    
    config = configs.get(sensitivity, configs['high'])
    return ImprovedPredictor(**config)


# Example usage functions
def predict_image_improved(efficientnet_model, swin_model, image_tensor_eff, 
                          image_tensor_swin, device, sensitivity='high'):
    """
    Improved image prediction with better fake detection
    
    Returns:
        dict with prediction results
    """
    predictor = create_improved_predictor(sensitivity)
    
    # Get predictions from both models
    eff_pred, eff_prob = predictor.enhanced_single_prediction(
        efficientnet_model, image_tensor_eff, device
    )
    
    swin_pred, swin_prob = predictor.enhanced_single_prediction(
        swin_model, image_tensor_swin, device
    )
    
    # Ensemble prediction
    predictions = [
        ('EfficientNet', eff_pred, eff_prob),
        ('Swin', swin_pred, swin_prob)
    ]
    
    final_pred, final_conf, details = predictor.ensemble_predict(predictions)
    
    return {
        'final_prediction': final_pred,
        'final_confidence': final_conf,
        'efficientnet': {'prediction': eff_pred, 'confidence': eff_prob},
        'swin': {'prediction': swin_pred, 'confidence': swin_prob},
        'details': details
    }


def predict_video_improved(frame_results, sensitivity='high', min_fake_ratio=0.25):
    """
    Improved video prediction with better fake detection
    
    Args:
        frame_results: List of frame-level predictions
        sensitivity: Detection sensitivity
        min_fake_ratio: Minimum fake frame ratio to classify video as fake
    """
    predictor = create_improved_predictor(sensitivity)
    
    # Aggregate frame predictions
    frame_predictions = []
    for frame_result in frame_results:
        # Extract prediction from frame result
        if isinstance(frame_result, dict):
            pred = frame_result.get('prediction', 'UNKNOWN')
        else:
            pred = frame_result
        frame_predictions.append(pred)
    
    final_pred, final_conf, details = predictor.aggregate_video_predictions(
        frame_predictions, min_fake_ratio
    )
    
    return {
        'final_prediction': final_pred,
        'final_confidence': final_conf,
        'details': details
    }


if __name__ == "__main__":
    print("Improved Predictor Module")
    print("="*60)
    print("\nFeatures:")
    print("  - Lower detection threshold (0.45 vs 0.5)")
    print("  - Aggressive ensemble mode")
    print("  - Confidence recalibration")
    print("  - Video-level aggregation with sensitivity")
    print("\nUsage:")
    print("  from utils_improved_predictor import predict_image_improved")
    print("  results = predict_image_improved(model1, model2, img1, img2, device)")
