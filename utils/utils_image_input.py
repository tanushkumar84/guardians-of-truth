import streamlit as st
import torch
from PIL import Image
import logging
from pathlib import Path
import time
from datetime import datetime
from .utils_image_processor import *
from .utils_model import get_cached_model
from .utils_format import format_confidence, format_prediction
from .utils_session import *
from .utils_improved_predictor import ImprovedPredictor, create_improved_predictor
from .utils_report_generator import ReportGenerator, create_downloadable_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image_input(uploaded_file):
    try:
        if uploaded_file is None:
            st.error("No file uploaded. Please select an image file.")
            return
            
        with cleanup_on_exit():
            start_time = time.time()
            
            try:
                image = Image.open(uploaded_file).convert('RGB')
            except Exception as e:
                st.error(f"Failed to load image. Please ensure the file is a valid image format (JPG, JPEG, PNG). Error: {str(e)}")
                logger.error(f"Image loading error: {str(e)}")
                return
            
            face_image, viz_image = extract_face(image)
            
            if face_image is None:
                st.error("No face detected in the image. Please upload an image containing a clear face.")
                return
            
            # Model predictions section (moved to the top)
            st.write("### Model Predictions")
            
            # Create improved predictor with high sensitivity
            predictor = create_improved_predictor(sensitivity='high')
            st.info("🔍 Using improved detection with HIGH sensitivity (threshold: 0.45, aggressive ensemble)")
            
            # Load models
            model_dir = Path("runs/models")
            
            # Check which models are available
            available_models = []
            available_models_display = []
            if (model_dir / "efficientnet/best_model_cpu.pth").exists():
                available_models.append("efficientnet")
                available_models_display.append("EfficientNet (Pre-trained)")
            if (model_dir / "swin/best_model_cpu.pth").exists():
                available_models.append("swin")
                available_models_display.append("Swin Transformer (Pre-trained)")
            if (model_dir / "custom_lightweight/best_model_cpu.pth").exists():
                available_models.append("custom")
                available_models_display.append("Custom CNN (81% Accuracy)")
            
            # Create columns based on available models
            num_cols = len(available_models) if available_models else 2
            cols = st.columns(num_cols)
            
            efficientnet_pred = None
            efficientnet_prob = None
            efficientnet_prob_real = None
            efficientnet_prob_fake = None
            swin_pred = None
            swin_prob = None
            swin_prob_real = None
            swin_prob_fake = None
            custom_pred = None
            custom_prob = None
            custom_prob_real = None
            custom_prob_fake = None
            device = torch.device('cpu')
            
            col_idx = 0

            # Process with EfficientNet
            if "efficientnet" in available_models:
                with cols[col_idx]:
                    efficientnet_model = get_cached_model(
                        model_dir / "efficientnet/best_model_cpu.pth", 
                        "efficientnet"
                    )
                    if efficientnet_model is not None:
                        processed_image = process_image(face_image, "efficientnet")
                        if processed_image is not None:
                            # Use improved predictor
                            efficientnet_pred, efficientnet_prob = predictor.enhanced_single_prediction(
                                efficientnet_model, processed_image, device
                            )
                            confidence = efficientnet_prob if efficientnet_pred == "FAKE" else 1 - efficientnet_prob
                            efficientnet_prob_fake = efficientnet_prob
                            efficientnet_prob_real = 1 - efficientnet_prob
                            
                            st.markdown(f"""
                            <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                                <h4>EFFICIENTNET</h4>
                                <p>Prediction: {format_prediction(efficientnet_pred)}<br>
                                Confidence: {format_confidence(confidence)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to process image for EfficientNet")
                col_idx += 1

            # Process with Swin
            if "swin" in available_models:
                with cols[col_idx]:
                    swin_model = get_cached_model(
                        model_dir / "swin/best_model_cpu.pth", 
                        "swin"
                    )
                    if swin_model is not None:
                        processed_image = process_image(face_image, "swin")
                        if processed_image is not None:
                            # Use improved predictor
                            swin_pred, swin_prob = predictor.enhanced_single_prediction(
                                swin_model, processed_image, device
                            )
                            confidence = swin_prob if swin_pred == "FAKE" else 1 - swin_prob
                            swin_prob_fake = swin_prob
                            swin_prob_real = 1 - swin_prob
                            
                            st.markdown(f"""
                            <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                                <h4>SWIN TRANSFORMER</h4>
                                <p>Prediction: {format_prediction(swin_pred)}<br>
                                Confidence: {format_confidence(confidence)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to process image for Swin")
                col_idx += 1
            
            # Process with Custom CNN
            if "custom" in available_models:
                with cols[col_idx]:
                    custom_model = get_cached_model(
                        model_dir / "custom_lightweight/best_model_cpu.pth", 
                        "custom"
                    )
                    if custom_model is not None:
                        processed_image = process_image(face_image, "custom")
                        if processed_image is not None:
                            custom_pred, custom_prob = predictor.enhanced_single_prediction(
                                custom_model, processed_image, device
                            )
                            confidence = custom_prob if custom_pred == "FAKE" else 1 - custom_prob
                            custom_prob_fake = custom_prob
                            custom_prob_real = 1 - custom_prob
                            
                            st.markdown(f"""
                            <div style='padding: 15px; border-radius: 10px; border: 1px solid #4CAF50; margin: 5px 0;'>
                                <h4>CUSTOM CNN 🚀</h4>
                                <p style='font-size: 0.9em; color: #4CAF50;'>⭐ Your Trained Model (81%)</p>
                                <p>Prediction: {format_prediction(custom_pred)}<br>
                                Confidence: {format_confidence(confidence)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to process image for Custom CNN")
                col_idx += 1
            
            # Use improved ensemble prediction
            if efficientnet_pred is None and swin_pred is None:
                st.error("No models loaded")
                return
            
            # Build predictions list for ensemble
            predictions = []
            if efficientnet_pred is not None and efficientnet_prob is not None:
                predictions.append(('EfficientNet', efficientnet_pred, efficientnet_prob))
            if swin_pred is not None and swin_prob is not None:
                predictions.append(('Swin', swin_pred, swin_prob))
            if "custom" in available_models and 'custom_pred' in locals() and custom_pred is not None:
                predictions.append(('Custom CNN', custom_pred, custom_prob))
            
            # Get ensemble prediction using improved logic
            overall_prediction, final_confidence, details = predictor.ensemble_predict(predictions)
            
            # Show ensemble details
            with st.expander("📊 Ensemble Details"):
                st.write(f"**Fake detections:** {details['fake_count']}/{details['total_models']}")
                st.write(f"**Average probability:** {details['avg_probability']:.4f}")
                st.write(f"**Max probability:** {details['max_probability']:.4f}")
                st.write(f"**Ensemble mode:** Aggressive (if any model detects fake with confidence > 0.55, classify as FAKE)")

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
            
            # Image section (moved below predictions)
            st.write("")  # Add some spacing
            
            # Use columns to align the images side by side with a gap
            col1, col2 = st.columns([1, 1])  # Equal width for both columns
            
            with col1:
                # Resize the original image for display
                resized_viz_image = resize_image_for_display(viz_image, max_size=500)  # Adjust max_size as needed
                st.image(resized_viz_image, caption="Original Image with Face Detection")  # Caption under the image
            
            with col2:
                # Resize the extracted face for display
                display_face = resize_image_for_display(face_image, max_size=500)  # Adjust max_size as needed
                st.image(display_face, caption="Extracted Face")  # Caption under the image
            
            # Calculate analysis time
            analysis_time = time.time() - start_time
            
            # Generate report data
            report_results = []
            if efficientnet_pred is not None:
                report_results.append({
                    'model_type': 'efficientnet',
                    'prediction': efficientnet_pred,
                    'confidence': efficientnet_prob if efficientnet_pred == "FAKE" else 1 - efficientnet_prob,
                    'probability_real': efficientnet_prob_real,
                    'probability_fake': efficientnet_prob_fake
                })
            if swin_pred is not None:
                report_results.append({
                    'model_type': 'swin',
                    'prediction': swin_pred,
                    'confidence': swin_prob if swin_pred == "FAKE" else 1 - swin_prob,
                    'probability_real': swin_prob_real,
                    'probability_fake': swin_prob_fake
                })
            if custom_pred is not None:
                report_results.append({
                    'model_type': 'custom',
                    'prediction': custom_pred,
                    'confidence': custom_prob if custom_pred == "FAKE" else 1 - custom_prob,
                    'probability_real': custom_prob_real,
                    'probability_fake': custom_prob_fake
                })
            
            # Create report generator
            report_gen = ReportGenerator()
            report_data = report_gen.generate_image_report(
                filename=uploaded_file.name,
                results=report_results,
                overall_prediction=overall_prediction,
                overall_confidence=final_confidence * 100,
                analysis_time=analysis_time,
                image_size=image.size
            )
            
            # Download section
            st.write("")
            st.write("### 📥 Download Analysis Report")
            st.info("Download a detailed report of this analysis with all metrics and interpretations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                txt_report = create_downloadable_report(report_data, format="txt")
                st.download_button(
                    label="📄 Download TXT Report",
                    data=txt_report,
                    file_name=f"deepfake_report_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                json_report = create_downloadable_report(report_data, format="json")
                st.download_button(
                    label="📊 Download JSON Report",
                    data=json_report,
                    file_name=f"deepfake_report_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                html_report = create_downloadable_report(report_data, format="html")
                st.download_button(
                    label="🌐 Download HTML Report",
                    data=html_report,
                    file_name=f"deepfake_report_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in process_image_input: {str(e)}")
        clear_session_data()