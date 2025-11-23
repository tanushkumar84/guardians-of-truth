"""
Full Frame Video Analysis
Analyzes entire video frames including objects, backgrounds, and environments
Not just faces - comprehensive scene-level deepfake detection
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FullFrameProcessor:
    """Process entire video frames for deepfake detection"""
    
    def __init__(self, 
                 frame_size: Tuple[int, int] = (224, 224),
                 enable_face_analysis: bool = True,
                 enable_scene_analysis: bool = True):
        """
        Args:
            frame_size: Size to resize frames to
            enable_face_analysis: Whether to include face-specific analysis
            enable_scene_analysis: Whether to analyze entire scene
        """
        self.frame_size = frame_size
        self.enable_face_analysis = enable_face_analysis
        self.enable_scene_analysis = enable_scene_analysis
    
    def preprocess_full_frame(self, frame: np.ndarray, model_type: str = 'efficientnet') -> torch.Tensor:
        """
        Preprocess entire frame for model input
        
        Args:
            frame: BGR frame from video
            model_type: 'efficientnet' or 'swin'
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_frame = Image.fromarray(frame_rgb)
        
        # Resize based on model
        if model_type == 'efficientnet':
            target_size = (300, 300)
        else:  # swin
            target_size = (224, 224)
        
        pil_frame = pil_frame.resize(target_size, Image.LANCZOS)
        
        # Convert to tensor and normalize
        frame_array = np.array(pil_frame).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        frame_normalized = (frame_array - mean) / std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def extract_frame_regions(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract different regions of the frame for analysis
        
        Returns:
            Dictionary with different frame regions:
            - full: Entire frame
            - center: Center region
            - top: Top region
            - bottom: Bottom region
            - left: Left region
            - right: Right region
        """
        h, w = frame.shape[:2]
        
        regions = {
            'full': frame,
            'center': frame[h//4:3*h//4, w//4:3*w//4],
            'top': frame[0:h//2, :],
            'bottom': frame[h//2:, :],
            'left': frame[:, 0:w//2],
            'right': frame[:, w//2:]
        }
        
        return regions
    
    def analyze_frame_statistics(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from frame
        
        Returns:
            Dictionary of frame statistics
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics
        stats = {
            # Color statistics
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'mean_saturation': np.mean(hsv[:, :, 1]),
            'std_saturation': np.std(hsv[:, :, 1]),
            
            # Texture statistics (using Laplacian)
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            
            # Edge statistics
            'edge_density': np.mean(cv2.Canny(gray, 100, 200)) / 255.0,
        }
        
        return stats
    
    def detect_artifacts(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect common deepfake artifacts in frame
        
        Returns:
            Dictionary of artifact scores
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        artifacts = {}
        
        # 1. Compression artifacts (JPEG blocking)
        # Use DCT to detect blocking artifacts
        dct = cv2.dct(np.float32(gray) / 255.0)
        artifacts['compression_artifact_score'] = np.std(dct)
        
        # 2. Blur inconsistency
        # Compare blur in different regions
        regions = self.extract_frame_regions(frame)
        blurs = []
        for region_name, region in regions.items():
            if region_name != 'full':
                region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(region_gray, cv2.CV_64F).var()
                blurs.append(blur)
        
        artifacts['blur_inconsistency'] = np.std(blurs) / (np.mean(blurs) + 1e-6)
        
        # 3. Color inconsistency
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Compare color distributions in different quadrants
        q1 = hsv[0:h//2, 0:w//2]
        q2 = hsv[0:h//2, w//2:]
        q3 = hsv[h//2:, 0:w//2]
        q4 = hsv[h//2:, w//2:]
        
        hue_stds = [np.std(q[:, :, 0]) for q in [q1, q2, q3, q4]]
        artifacts['color_inconsistency'] = np.std(hue_stds)
        
        # 4. Lighting inconsistency
        # Analyze lighting distribution
        brightness_stds = [np.std(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)) 
                          for q in [q1, q2, q3, q4]]
        artifacts['lighting_inconsistency'] = np.std(brightness_stds)
        
        return artifacts


class MultiRegionPredictor:
    """Predict on multiple regions and aggregate results"""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 frame_processor: FullFrameProcessor):
        self.model = model
        self.device = device
        self.frame_processor = frame_processor
        self.model.eval()
    
    def predict_full_frame(self, frame: np.ndarray, model_type: str) -> Dict:
        """
        Predict on entire frame
        
        Returns:
            Dictionary with prediction and confidence
        """
        # Preprocess full frame
        frame_tensor = self.frame_processor.preprocess_full_frame(frame, model_type)
        frame_tensor = frame_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(frame_tensor)
            if isinstance(output, tuple):
                output = output[0]
            
            # Handle both single output (sigmoid) and dual output (softmax)
            if output.numel() == 1:
                # Single output - use sigmoid
                probability = torch.sigmoid(output).item()
            else:
                # Dual output (2 classes) - use softmax
                probabilities = torch.softmax(output, dim=1)
                probability = probabilities[0, 1].item()  # Probability of FAKE class
        
        prediction = "FAKE" if probability > 0.45 else "REAL"
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': probability if prediction == "FAKE" else 1 - probability
        }
    
    def predict_multi_region(self, frame: np.ndarray, model_type: str) -> Dict:
        """
        Predict on multiple frame regions and aggregate
        
        Returns:
            Dictionary with aggregated prediction
        """
        # Extract regions
        regions = self.frame_processor.extract_frame_regions(frame)
        
        # Predict on each region
        region_predictions = {}
        probabilities = []
        
        for region_name, region_frame in regions.items():
            # Skip if region too small
            if region_frame.shape[0] < 50 or region_frame.shape[1] < 50:
                continue
            
            result = self.predict_full_frame(region_frame, model_type)
            region_predictions[region_name] = result
            probabilities.append(result['probability'])
        
        # Aggregate predictions
        avg_probability = np.mean(probabilities)
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        std_probability = np.std(probabilities)
        
        # Use max probability for final prediction (most suspicious region)
        final_prediction = "FAKE" if max_probability > 0.45 else "REAL"
        
        return {
            'prediction': final_prediction,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'min_probability': min_probability,
            'std_probability': std_probability,
            'region_predictions': region_predictions,
            'inconsistency_score': std_probability  # High std = inconsistent regions
        }
    
    def predict_with_artifacts(self, frame: np.ndarray, model_type: str) -> Dict:
        """
        Predict with artifact analysis
        
        Returns:
            Comprehensive prediction with artifacts
        """
        # Get multi-region prediction
        region_result = self.predict_multi_region(frame, model_type)
        
        # Get frame statistics
        stats = self.frame_processor.analyze_frame_statistics(frame)
        
        # Get artifact scores
        artifacts = self.frame_processor.detect_artifacts(frame)
        
        # Calculate artifact-based suspicion score
        artifact_score = (
            artifacts['blur_inconsistency'] * 0.3 +
            artifacts['color_inconsistency'] * 0.3 +
            artifacts['lighting_inconsistency'] * 0.4
        )
        
        # Normalize artifact score (0-1)
        artifact_score = min(1.0, artifact_score / 10.0)
        
        # Combine model prediction with artifacts
        model_prob = region_result['max_probability']
        combined_prob = 0.7 * model_prob + 0.3 * artifact_score
        
        final_prediction = "FAKE" if combined_prob > 0.45 else "REAL"
        
        return {
            'prediction': final_prediction,
            'combined_probability': combined_prob,
            'model_probability': model_prob,
            'artifact_score': artifact_score,
            'confidence': combined_prob if final_prediction == "FAKE" else 1 - combined_prob,
            'region_analysis': region_result,
            'frame_statistics': stats,
            'detected_artifacts': artifacts
        }


def analyze_video_full_frame(video_frames: List[np.ndarray], 
                             models: List[Dict],
                             device: torch.device,
                             analyze_artifacts: bool = True) -> Dict:
    """
    Analyze entire video using full frame analysis
    
    Args:
        video_frames: List of video frames (BGR format)
        models: List of model dictionaries with 'model' and 'model_type'
        device: Device to run on
        analyze_artifacts: Whether to include artifact analysis
        
    Returns:
        Comprehensive video analysis results
    """
    frame_processor = FullFrameProcessor()
    
    all_results = []
    
    for frame_idx, frame in enumerate(video_frames):
        frame_results = []
        
        for model_data in models:
            predictor = MultiRegionPredictor(
                model_data['model'],
                device,
                frame_processor
            )
            
            if analyze_artifacts:
                result = predictor.predict_with_artifacts(frame, model_data['model_type'])
            else:
                result = predictor.predict_multi_region(frame, model_data['model_type'])
            
            result['model_type'] = model_data['model_type']
            result['frame_index'] = frame_idx
            frame_results.append(result)
        
        all_results.append(frame_results)
    
    # Aggregate across all frames
    video_analysis = aggregate_video_results(all_results)
    
    return video_analysis


def aggregate_video_results(frame_results: List[List[Dict]]) -> Dict:
    """
    Aggregate frame-level results into video-level prediction
    
    Args:
        frame_results: List of frame results, each containing model predictions
        
    Returns:
        Video-level analysis
    """
    # Collect predictions per model
    model_predictions = {}
    
    for frame_result_list in frame_results:
        for result in frame_result_list:
            model_type = result['model_type']
            
            if model_type not in model_predictions:
                model_predictions[model_type] = {
                    'predictions': [],
                    'probabilities': [],
                    'artifact_scores': []
                }
            
            model_predictions[model_type]['predictions'].append(result['prediction'])
            
            # Use combined probability if available
            if 'combined_probability' in result:
                prob = result['combined_probability']
                model_predictions[model_type]['probabilities'].append(prob)
                model_predictions[model_type]['artifact_scores'].append(result['artifact_score'])
            else:
                prob = result['max_probability']
                model_predictions[model_type]['probabilities'].append(prob)
    
    # Calculate per-model video predictions
    video_predictions = []
    
    for model_type, data in model_predictions.items():
        fake_count = sum(1 for p in data['predictions'] if p == "FAKE")
        total_frames = len(data['predictions'])
        fake_ratio = fake_count / total_frames
        
        avg_probability = np.mean(data['probabilities'])
        max_probability = np.max(data['probabilities'])
        
        # Aggressive: 25% fake frames = fake video
        video_prediction = "FAKE" if fake_ratio >= 0.25 else "REAL"
        video_confidence = avg_probability if video_prediction == "FAKE" else 1 - avg_probability
        
        avg_artifact_score = np.mean(data['artifact_scores']) if data['artifact_scores'] else 0.0
        
        video_predictions.append({
            'model_type': model_type,
            'prediction': video_prediction,
            'confidence': video_confidence,
            'fake_frame_ratio': fake_ratio,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'avg_artifact_score': avg_artifact_score,
            'fake_frames': fake_count,
            'total_frames': total_frames
        })
    
    # Ensemble decision
    fake_models = sum(1 for vp in video_predictions if vp['prediction'] == "FAKE")
    final_prediction = "FAKE" if fake_models > 0 else "REAL"
    
    # Calculate ensemble confidence
    all_probs = [vp['avg_probability'] for vp in video_predictions]
    ensemble_confidence = np.mean(all_probs)
    
    if final_prediction == "REAL":
        ensemble_confidence = 1 - ensemble_confidence
    
    return {
        'final_prediction': final_prediction,
        'ensemble_confidence': ensemble_confidence,
        'models_detecting_fake': fake_models,
        'total_models': len(video_predictions),
        'model_predictions': video_predictions,
        'analysis_type': 'full_frame',
        'total_frames_analyzed': len(frame_results)
    }


if __name__ == "__main__":
    print("Full Frame Video Analysis Module")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Analyzes ENTIRE video frames (not just faces)")
    print("  ✓ Multi-region analysis (center, edges, quadrants)")
    print("  ✓ Artifact detection (compression, blur, color, lighting)")
    print("  ✓ Statistical analysis (brightness, saturation, sharpness)")
    print("  ✓ Inconsistency detection across frame regions")
    print("\nUsage:")
    print("  from utils_full_frame_analysis import analyze_video_full_frame")
    print("  results = analyze_video_full_frame(frames, models, device)")
