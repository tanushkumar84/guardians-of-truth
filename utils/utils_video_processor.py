import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, List
import tempfile
import logging
from .utils_improved_predictor import create_improved_predictor
from .utils_full_frame_analysis import (
    FullFrameProcessor, 
    MultiRegionPredictor,
    analyze_video_full_frame
)

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, num_frames: int = 100, sensitivity: str = 'high', 
                 analyze_full_frame: bool = True, analyze_artifacts: bool = True):
        """
        Initialize video processor
        num_frames: Number of frames to extract from video
        sensitivity: Detection sensitivity ('low', 'medium', 'high', 'very_high')
        analyze_full_frame: Whether to analyze entire frame (not just face)
        analyze_artifacts: Whether to detect deepfake artifacts
        """
        self.num_frames = num_frames
        self.predictor = create_improved_predictor(sensitivity=sensitivity)
        self.analyze_full_frame = analyze_full_frame
        self.analyze_artifacts = analyze_artifacts
        self.frame_processor = FullFrameProcessor() if analyze_full_frame else None
    
    def save_uploaded_video(self, video_file) -> str:
        """Save uploaded video file to temporary location"""
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            return tfile.name
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, progress_callback=None) -> List[np.ndarray]:
        """Extract evenly spaced frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to get desired number of frames
            frame_interval = max(1, total_frames // self.num_frames)
            
            frames = []
            frame_count = 0
            
            while cap.isOpened() and len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    if progress_callback:
                        progress = len(frames) / self.num_frames
                        progress_callback(progress)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def process_video(self, video_path: str, extract_face_fn, process_image_fn, models: List[dict], 
                     progress_callbacks: dict = None) -> Tuple[List[dict], List[List[dict]], List[Image.Image]]:
        """Process video frames and return results"""
        try:
            # Initialize progress callbacks
            if progress_callbacks is None:
                progress_callbacks = {
                    'extract_frames': lambda x: None,
                    'extract_faces': lambda x: None,
                    'process_frames': lambda x: None
                }
            
            # Extract frames
            frames = self.extract_frames(video_path, progress_callbacks['extract_frames'])
            
            # Process frames
            results = []  # Final aggregated results
            frame_results = []  # Results for each frame
            faces = []  # Store detected faces
            processed_frames = 0
            total_frames = len(frames)
            
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Extract face using the same function as images
                face_image, _ = extract_face_fn(frame_pil)
                processed_frames += 1
                progress_callbacks['extract_faces'](processed_frames / total_frames)
                
                if face_image is not None:
                    faces.append(face_image)  # Store the face
                    # Process face through models
                    frame_results_current = []
                    for model_data in models:
                        # Process image using the same function as images
                        processed_image = process_image_fn(face_image, model_data['model_type'])
                        if processed_image is not None:
                            # Make prediction using improved predictor
                            device = torch.device('cpu')
                            prediction, probability = self.predictor.enhanced_single_prediction(
                                model_data['model'], processed_image, device
                            )
                            confidence = probability if prediction == "FAKE" else 1 - probability
                            
                            frame_results_current.append({
                                'model_type': model_data['model_type'],
                                'prediction': prediction,
                                'confidence': confidence,
                                'probability': probability
                            })
                    
                    if frame_results_current:
                        frame_results.append(frame_results_current)
                
                progress_callbacks['process_frames'](processed_frames / total_frames)
            
            # Calculate average results across all frames using improved aggregation
            final_results = []
            if frame_results:
                # Get unique model types
                model_types = {result['model_type'] for frame_result in frame_results for result in frame_result}
                
                for model_type in model_types:
                    # Get all predictions for this model
                    model_predictions = [
                        result for frame_result in frame_results
                        for result in frame_result
                        if result['model_type'] == model_type
                    ]
                    
                    # Calculate statistics
                    total_predictions = len(model_predictions)
                    fake_count = sum(1 for p in model_predictions if p['prediction'] == "FAKE")
                    fake_ratio = fake_count / total_predictions
                    avg_confidence = sum(p['confidence'] for p in model_predictions) / total_predictions
                    
                    # Use improved video aggregation logic (25% fake frames = fake video)
                    frame_pred_list = [p['prediction'] for p in model_predictions]
                    video_pred, video_conf, details = self.predictor.aggregate_video_predictions(
                        frame_pred_list, min_fake_ratio=0.25
                    )
                    
                    final_results.append({
                        'model_type': model_type,
                        'prediction': video_pred,
                        'confidence': video_conf,
                        'fake_frame_ratio': fake_ratio,
                        'fake_count': fake_count,
                        'total_frames': total_predictions
                    })
            
            return final_results, frame_results, faces
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    def process_video_full_frame(self, video_path: str, models: List[dict],
                                progress_callbacks: dict = None) -> Tuple[dict, List[np.ndarray]]:
        """
        Process video with FULL FRAME analysis (not just faces)
        Analyzes entire frames including objects, backgrounds, and environments
        
        Args:
            video_path: Path to video file
            models: List of model dictionaries with 'model' and 'model_type'
            progress_callbacks: Progress update callbacks
            
        Returns:
            (results_dict, frames_list)
        """
        try:
            # Initialize progress callbacks
            if progress_callbacks is None:
                progress_callbacks = {
                    'extract_frames': lambda x: None,
                    'process_frames': lambda x: None
                }
            
            # Extract frames
            logger.info(f"Extracting {self.num_frames} frames for full frame analysis...")
            frames = self.extract_frames(video_path, progress_callbacks['extract_frames'])
            logger.info(f"Extracted {len(frames)} frames")
            
            # Process frames with full frame analysis
            logger.info("Analyzing entire frames (not just faces)...")
            
            device = torch.device('cpu')
            all_frame_results = []
            
            for frame_idx, frame in enumerate(frames):
                frame_results = []
                
                for model_data in models:
                    # Create multi-region predictor
                    predictor = MultiRegionPredictor(
                        model_data['model'],
                        device,
                        self.frame_processor
                    )
                    
                    # Analyze frame with artifacts
                    if self.analyze_artifacts:
                        result = predictor.predict_with_artifacts(
                            frame, model_data['model_type']
                        )
                    else:
                        result = predictor.predict_multi_region(
                            frame, model_data['model_type']
                        )
                    
                    result['model_type'] = model_data['model_type']
                    result['frame_index'] = frame_idx
                    frame_results.append(result)
                
                all_frame_results.append(frame_results)
                
                # Update progress
                progress = (frame_idx + 1) / len(frames)
                progress_callbacks['process_frames'](progress)
            
            # Aggregate results across all frames
            logger.info("Aggregating frame results...")
            video_analysis = self.aggregate_full_frame_results(all_frame_results)
            
            return video_analysis, frames
            
        except Exception as e:
            logger.error(f"Error in full frame video processing: {str(e)}")
            raise
    
    def aggregate_full_frame_results(self, frame_results: List[List[dict]]) -> dict:
        """
        Aggregate frame-level full frame results into video-level prediction
        
        Args:
            frame_results: List of frame results
            
        Returns:
            Video-level analysis dictionary
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
                        'artifact_scores': [],
                        'region_inconsistencies': []
                    }
                
                model_predictions[model_type]['predictions'].append(result['prediction'])
                
                # Use combined probability if available (includes artifacts)
                if 'combined_probability' in result:
                    prob = result['combined_probability']
                    model_predictions[model_type]['probabilities'].append(prob)
                    model_predictions[model_type]['artifact_scores'].append(result['artifact_score'])
                    
                    # Track region inconsistency
                    if 'region_analysis' in result:
                        inconsistency = result['region_analysis'].get('inconsistency_score', 0)
                        model_predictions[model_type]['region_inconsistencies'].append(inconsistency)
                else:
                    prob = result.get('max_probability', result.get('avg_probability', 0.5))
                    model_predictions[model_type]['probabilities'].append(prob)
        
        # Calculate per-model video predictions
        final_results = []
        
        for model_type, data in model_predictions.items():
            fake_count = sum(1 for p in data['predictions'] if p == "FAKE")
            total_frames = len(data['predictions'])
            fake_ratio = fake_count / total_frames
            
            avg_probability = np.mean(data['probabilities'])
            max_probability = np.max(data['probabilities'])
            
            # Aggressive: 25% fake frames = fake video
            video_prediction = "FAKE" if fake_ratio >= 0.25 else "REAL"
            video_confidence = avg_probability if video_prediction == "FAKE" else 1 - avg_probability
            
            # Calculate average artifact score
            avg_artifact_score = np.mean(data['artifact_scores']) if data['artifact_scores'] else 0.0
            avg_inconsistency = np.mean(data['region_inconsistencies']) if data['region_inconsistencies'] else 0.0
            
            final_results.append({
                'model_type': model_type,
                'prediction': video_prediction,
                'confidence': video_confidence,
                'fake_frame_ratio': fake_ratio,
                'fake_count': fake_count,
                'total_frames': total_frames,
                'avg_probability': avg_probability,
                'max_probability': max_probability,
                'avg_artifact_score': avg_artifact_score,
                'avg_region_inconsistency': avg_inconsistency,
                'analysis_type': 'full_frame_with_artifacts' if self.analyze_artifacts else 'full_frame'
            })
        
        # Ensemble decision
        fake_models = sum(1 for r in final_results if r['prediction'] == "FAKE")
        
        return {
            'results': final_results,
            'models_detecting_fake': fake_models,
            'total_models': len(final_results),
            'analysis_type': 'full_frame'
        } 