import streamlit as st
import os
import logging
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from .utils_video_processor import VideoProcessor
from pathlib import Path
from .utils_image_processor import extract_face, process_image, resize_image_for_display
from .utils_model import get_cached_model
from .utils_format import format_confidence, format_prediction
from .utils_improved_predictor import create_improved_predictor
from .utils_report_generator import ReportGenerator, create_downloadable_report
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_IMAGE_SIZES = {
    "efficientnet": 300,
    "swin": 224,
    "custom": 128
}

def process_video_input(video_file):
    try:
        video_path = None  # Initialize video_path to None
        # Initialize session state for processing
        if 'processing_started' not in st.session_state:
            st.session_state.processing_started = False
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'faces' not in st.session_state:
            st.session_state.faces = None

        # Only show the slider and button if processing hasn't started
        if not st.session_state.processing_started:
            # Analysis mode selection
            st.write("### Analysis Settings")
            
            analysis_mode = st.radio(
                "Analysis Mode:",
                ["Full Frame Analysis (Entire Video)", "Face-Only Analysis"],
                index=0,
                help="Full Frame: Analyzes entire frames including objects, backgrounds, and environments. Face-Only: Analyzes only detected faces."
            )
            
            if analysis_mode == "Full Frame Analysis (Entire Video)":
                st.info("🎬 **Full Frame Mode**: Analyzes entire video frames including:\n- Objects and backgrounds\n- Scene consistency\n- Compression artifacts\n- Lighting and color inconsistencies\n- Blur patterns across regions")
                analyze_full_frame = True
            else:
                st.info("👤 **Face-Only Mode**: Traditional analysis focusing on detected faces")
                analyze_full_frame = False
            
            # Get number of frames from user
            col1, col2 = st.columns([3, 1])
            with col1:
                num_frames = st.slider("Number of frames to analyze", min_value=10, max_value=300, value=30, step=10,
                                    help="More frames = more accurate but slower processing")
            with col2:
                if st.button("Start Processing", type="primary"):
                    st.session_state.processing_started = True
                    st.session_state.num_frames = num_frames
                    st.session_state.analyze_full_frame = analyze_full_frame
                    st.rerun()
            return

        # If processing has started but not complete, show the progress bar
        if st.session_state.processing_started and not st.session_state.processing_complete:
            # Create a single progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Get analysis mode
            analyze_full_frame = st.session_state.get('analyze_full_frame', True)
            
            # Initialize video processor with high sensitivity
            video_processor = VideoProcessor(
                num_frames=st.session_state.num_frames,
                sensitivity='high',
                analyze_full_frame=analyze_full_frame,
                analyze_artifacts=True
            )
            video_path = video_processor.save_uploaded_video(video_file)
            
            # Show analysis mode info
            if analyze_full_frame:
                st.success("🎬 **FULL FRAME ANALYSIS MODE**")
                st.info("Analyzing entire video frames including objects, backgrounds, artifacts, and scene inconsistencies")
            else:
                st.info("� Face-only analysis mode")
            
            st.info("🔍 Detection: HIGH sensitivity | Threshold: 0.45 | 25% fake frames = FAKE video")
            
            # Load models
            status_text.text("🔄 Loading models...")
            progress_bar.progress(0.2)
            
            model_dir = Path("runs/models")
            models_data = []
            
            # Load models (EfficientNet, Swin, and Custom CNN)
            efficientnet_model = get_cached_model(model_dir / "efficientnet/best_model_cpu.pth", "efficientnet")
            if efficientnet_model is not None:
                models_data.append({
                    'model': efficientnet_model,
                    'model_type': 'efficientnet',
                    'image_size': MODEL_IMAGE_SIZES['efficientnet']
                })
            
            swin_model = get_cached_model(model_dir / "swin/best_model_cpu.pth", "swin")
            if swin_model is not None:
                models_data.append({
                    'model': swin_model,
                    'model_type': 'swin',
                    'image_size': MODEL_IMAGE_SIZES['swin']
                })
            
            custom_model = get_cached_model(model_dir / "custom_lightweight/best_model_cpu.pth", "custom")
            if custom_model is not None:
                models_data.append({
                    'model': custom_model,
                    'model_type': 'custom',
                    'image_size': MODEL_IMAGE_SIZES['custom']
                })
            
            if not models_data:
                st.error("No models could be loaded! Please check the model files.")
                return
            
            # Define progress callback
            total_progress = {'value': 0}  # Use dictionary to maintain state
            
            def update_progress(stage):
                def callback(progress):
                    # Update total progress based on stage
                    if stage == 'extract_frames':
                        total_progress['value'] = progress * 0.5  # First half
                    elif stage in ['extract_faces', 'process_frames']:
                        total_progress['value'] = 0.5 + (progress * 0.5)  # Second half
                    
                    # Ensure progress only moves forward
                    progress_bar.progress(total_progress['value'])
                    status_text.text(f"🎥 Processing video... {int(total_progress['value'] * 100)}%")
                return callback
            
            progress_callbacks = {
                'extract_frames': update_progress('extract_frames'),
                'extract_faces': update_progress('extract_faces'),
                'process_frames': update_progress('process_frames')
            }
            
            # Process video based on analysis mode
            import time
            start_time = time.time()
            
            if analyze_full_frame:
                # FULL FRAME ANALYSIS - Analyze entire frames
                status_text.text("🎬 Analyzing entire video frames...")
                video_analysis, frames = video_processor.process_video_full_frame(
                    video_path,
                    models=models_data,
                    progress_callbacks=progress_callbacks
                )
                
                # Convert to standard format
                results = video_analysis.get('results', [])
                faces = []  # No face extraction in full frame mode
                frame_results = []
                
                st.session_state.analysis_mode = 'full_frame'
            else:
                # FACE-ONLY ANALYSIS - Traditional method
                status_text.text("👤 Analyzing detected faces...")
                results, frame_results, faces = video_processor.process_video(
                    video_path,
                    extract_face_fn=extract_face,
                    process_image_fn=process_image,
                    models=models_data,
                    progress_callbacks=progress_callbacks
                )
                st.session_state.analysis_mode = 'face_only'
            
            # Calculate analysis time
            analysis_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.faces = faces
            st.session_state.analysis_time = analysis_time
            st.session_state.frames = frames if analyze_full_frame else None
            
            # Clear progress indicators
            progress_container.empty()
            st.session_state.processing_complete = True
            st.rerun()

        # If processing is complete, show the results
        if st.session_state.processing_complete:
            results = st.session_state.results
            faces = st.session_state.faces

            if results:
                # Show analysis mode
                analysis_mode = st.session_state.get('analysis_mode', 'face_only')
                if analysis_mode == 'full_frame':
                    st.success("✅ **Full Frame Analysis Complete** - Analyzed entire video frames")
                else:
                    st.info("✅ **Face Analysis Complete** - Analyzed detected faces")
                
                st.write("### Model Predictions")
                cols = st.columns(3)  # Changed from 2 to 3 columns for 3 models
                
                overall_prediction = "REAL"  # Default to REAL
                for idx, result in enumerate(results):
                    with cols[idx % 3]:  # Changed from 2 to 3
                        model_name = result['model_type'].upper()
                        
                        # Build prediction display
                        prediction_html = f"""<div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'><h4>{model_name}</h4><p>Overall Prediction: {format_prediction(result['prediction'])}<br>Confidence: {format_confidence(result['confidence'])}<br>Fake Frames: {result['fake_frame_ratio']:.1%} ({result['fake_count']}/{result['total_frames']})</p>"""
                        
                        # Add full frame analysis details if available
                        if analysis_mode == 'full_frame':
                            avg_artifact = result.get('avg_artifact_score', 0)
                            avg_inconsistency = result.get('avg_region_inconsistency', 0)
                            prediction_html += f"""<div style='font-size: 0.9em; color: #666; margin-top: 8px; border-top: 1px solid #eee; padding-top: 8px;'><strong>Full Frame Analysis:</strong><br>📊 Artifact Score: {avg_artifact:.3f}<br>🔍 Region Inconsistency: {avg_inconsistency:.3f}<br>🎬 Analysis: Entire frames (objects, backgrounds, scenes)</div>"""
                        
                        prediction_html += "</div>"
                        st.markdown(prediction_html, unsafe_allow_html=True)
                        
                        # Determine overall prediction
                        if result['prediction'] == "FAKE":
                            overall_prediction = "FAKE"
                
                # Display overall verdict with improved styling
                st.markdown(f"""
                <div style='
                    background-color: {"rgba(255, 68, 68, 0.1)" if overall_prediction == "FAKE" else "rgba(0, 255, 157, 0.1)"};
                    border: 3px solid {"#ff4444" if overall_prediction == "FAKE" else "#00ff9d"};
                    border-radius: 15px;
                    padding: 20px;
                    margin: 30px 0;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                '>
                    <div style='
                        font-size: 4em;
                        margin: 0;
                        color: {"#ff4444" if overall_prediction == "FAKE" else "#00ff9d"};
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
                        font-weight: bold;
                    '>
                        {overall_prediction}
                    </div>
                    <p style='
                        font-size: 1.2em;
                        margin: 10px 0 0 0;
                        color: {"#ff4444" if overall_prediction == "FAKE" else "#00ff9d"};
                    '>
                        Final Verdict
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display samples based on analysis mode
                if analysis_mode == 'full_frame':
                    # Show sample full frames
                    st.write("### Sample Analyzed Frames")
                    st.info("Below are sample frames that were analyzed in their entirety (not just faces)")
                    
                    frames = st.session_state.get('frames', [])
                    if frames:
                        n_sample_frames = min(12, len(frames))
                        sample_indices = np.linspace(0, len(frames)-1, n_sample_frames, dtype=int)
                        
                        cols = st.columns(4)
                        for idx, frame_idx in enumerate(sample_indices):
                            with cols[idx % 4]:
                                frame = frames[int(frame_idx)]
                                # Convert BGR to RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_pil = Image.fromarray(frame_rgb)
                                resized_frame = resize_image_for_display(frame_pil, max_size=200)
                                st.markdown('<div class="face-grid-image">', unsafe_allow_html=True)
                                st.image(resized_frame, caption=f"Frame {int(frame_idx)+1}", width='content')
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No frames available for display")
                else:
                    # Show detected faces
                    st.write("### Sample Detected Faces")
                    if faces:
                        n_sample_faces = min(12, len(faces))
                        sample_indices = np.linspace(0, len(faces)-1, n_sample_faces, dtype=int)
                        
                        cols = st.columns(4)
                        for idx, face_idx in enumerate(sample_indices):
                            with cols[idx % 4]:
                                face = faces[int(face_idx)]
                                resized_face = resize_image_for_display(face, max_size=200)
                                st.markdown('<div class="face-grid-image">', unsafe_allow_html=True)
                                st.image(resized_face, caption=f"Face {int(face_idx)+1}", width='content')
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No faces could be detected in the video frames.")
            
            # Generate and offer report downloads
            if st.session_state.processing_complete and results:
                st.write("")
                st.write("### 📥 Download Analysis Report")
                st.info("Download a detailed report of this video analysis with all metrics and interpretations")
                
                # Get analysis metadata
                analyze_full_frame = st.session_state.get('analyze_full_frame', False)
                analysis_mode_str = 'full_frame' if analyze_full_frame else 'face_only'
                num_frames = st.session_state.get('num_frames', 30)
                analysis_time = st.session_state.get('analysis_time', 0)
                
                # Calculate overall prediction and confidence
                overall_prediction = "REAL"
                overall_confidence = 0.0
                fake_count = 0
                
                for result in results:
                    if result['prediction'] == "FAKE":
                        overall_prediction = "FAKE"
                        fake_count += 1
                    overall_confidence = max(overall_confidence, result.get('confidence', 0))
                
                # Prepare results for report
                report_results = []
                for result in results:
                    # Calculate avg_prob_real and avg_prob_fake from the available data
                    avg_probability = result.get('avg_probability', result.get('confidence', 0))
                    
                    # If prediction is FAKE, avg_probability represents probability of FAKE
                    # If prediction is REAL, avg_probability represents probability of REAL
                    if result.get('prediction') == 'FAKE':
                        avg_prob_fake = avg_probability
                        avg_prob_real = 1.0 - avg_probability
                    else:
                        avg_prob_real = avg_probability
                        avg_prob_fake = 1.0 - avg_probability
                    
                    report_result = {
                        'model_type': result.get('model_type', 'unknown'),
                        'prediction': result.get('prediction', 'Unknown'),
                        'confidence': result.get('confidence', 0),
                        'num_faces': result.get('total_frames', result.get('num_faces', num_frames)),
                        'fake_count': result.get('fake_count', 0),
                        'fake_ratio': result.get('fake_frame_ratio', result.get('fake_ratio', 0)),
                        'avg_prob_real': avg_prob_real,
                        'avg_prob_fake': avg_prob_fake
                    }
                    # Add artifact scores if available
                    if 'avg_artifact_score' in result:
                        report_result['avg_artifact_score'] = result['avg_artifact_score']
                        report_result['avg_region_inconsistency'] = result.get('avg_region_inconsistency', 0)
                    report_results.append(report_result)
                
                # Create report generator
                report_gen = ReportGenerator()
                report_data = report_gen.generate_video_report(
                    filename=video_file.name,
                    results=report_results,
                    overall_prediction=overall_prediction,
                    overall_confidence=overall_confidence * 100,
                    analysis_time=analysis_time,
                    num_frames_analyzed=num_frames,
                    video_duration=None,  # Could extract from video metadata
                    analysis_mode=analysis_mode_str
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    txt_report = create_downloadable_report(report_data, format="txt")
                    st.download_button(
                        label="📄 Download TXT Report",
                        data=txt_report,
                        file_name=f"deepfake_video_report_{video_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    json_report = create_downloadable_report(report_data, format="json")
                    st.download_button(
                        label="📊 Download JSON Report",
                        data=json_report,
                        file_name=f"deepfake_video_report_{video_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    html_report = create_downloadable_report(report_data, format="html")
                    st.download_button(
                        label="🌐 Download HTML Report",
                        data=html_report,
                        file_name=f"deepfake_video_report_{video_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            
            # Cleanup
            if video_path is not None and os.path.exists(video_path):
                os.remove(video_path)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")