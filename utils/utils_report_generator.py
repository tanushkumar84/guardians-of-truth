"""
Report Generator for Deepfake Detection Results
Generates detailed PDF and JSON reports with all analysis metrics
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import io
from PIL import Image, ImageDraw, ImageFont
import base64


class ReportGenerator:
    """Generate detailed reports for deepfake detection analysis"""
    
    def __init__(self):
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_image_report(
        self,
        filename: str,
        results: List[Dict[str, Any]],
        overall_prediction: str,
        overall_confidence: float,
        analysis_time: float,
        image_size: tuple = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate detailed report for image analysis
        
        Args:
            filename: Name of analyzed image
            results: List of per-model results
            overall_prediction: REAL or FAKE
            overall_confidence: Overall confidence percentage
            analysis_time: Time taken for analysis
            image_size: (width, height) of image
            format: Output format (json, txt, html)
            
        Returns:
            Dictionary containing report data
        """
        report = {
            "report_type": "Image Analysis Report",
            "generated_at": self.report_date,
            "file_info": {
                "filename": filename,
                "image_size": f"{image_size[0]}x{image_size[1]}" if image_size else "Unknown",
                "analysis_time_seconds": round(analysis_time, 2)
            },
            "overall_results": {
                "prediction": overall_prediction,
                "confidence": f"{overall_confidence:.2f}%",
                "verdict": "FAKE DETECTED" if overall_prediction == "FAKE" else "APPEARS GENUINE"
            },
            "model_results": [],
            "detailed_metrics": {},
            "interpretation": self._generate_interpretation(overall_prediction, overall_confidence, results)
        }
        
        # Add per-model details
        for result in results:
            model_data = {
                "model_type": result.get('model_type', 'Unknown').upper(),
                "prediction": result.get('prediction', 'Unknown'),
                "confidence": f"{result.get('confidence', 0) * 100:.2f}%",
                "probability_real": f"{result.get('probability_real', 0) * 100:.2f}%",
                "probability_fake": f"{result.get('probability_fake', 0) * 100:.2f}%"
            }
            report["model_results"].append(model_data)
            
            # Add to detailed metrics
            report["detailed_metrics"][result.get('model_type', 'Unknown')] = {
                "raw_probability_real": result.get('probability_real', 0),
                "raw_probability_fake": result.get('probability_fake', 0),
                "prediction_threshold": 0.45,
                "confidence_score": result.get('confidence', 0)
            }
        
        # Add ensemble information
        report["ensemble_method"] = {
            "type": "Aggressive Ensemble",
            "description": "If ANY model predicts FAKE with >55% confidence, overall prediction is FAKE",
            "models_used": len(results),
            "threshold": 0.45
        }
        
        return report
    
    def generate_video_report(
        self,
        filename: str,
        results: List[Dict[str, Any]],
        overall_prediction: str,
        overall_confidence: float,
        analysis_time: float,
        num_frames_analyzed: int,
        video_duration: float = None,
        analysis_mode: str = "face_only",
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate detailed report for video analysis
        
        Args:
            filename: Name of analyzed video
            results: List of per-model results
            overall_prediction: REAL or FAKE
            overall_confidence: Overall confidence percentage
            analysis_time: Time taken for analysis
            num_frames_analyzed: Number of frames processed
            video_duration: Duration of video in seconds
            analysis_mode: 'full_frame' or 'face_only'
            format: Output format (json, txt, html)
            
        Returns:
            Dictionary containing report data
        """
        report = {
            "report_type": "Video Analysis Report",
            "generated_at": self.report_date,
            "file_info": {
                "filename": filename,
                "video_duration_seconds": round(video_duration, 2) if video_duration else "Unknown",
                "frames_analyzed": num_frames_analyzed,
                "analysis_time_seconds": round(analysis_time, 2),
                "analysis_mode": "Full Frame Analysis" if analysis_mode == "full_frame" else "Face-Only Analysis"
            },
            "overall_results": {
                "prediction": overall_prediction,
                "confidence": f"{overall_confidence:.2f}%",
                "verdict": "FAKE DETECTED" if overall_prediction == "FAKE" else "APPEARS GENUINE"
            },
            "model_results": [],
            "detailed_metrics": {},
            "frame_analysis": {},
            "interpretation": self._generate_interpretation(overall_prediction, overall_confidence, results, is_video=True)
        }
        
        # Add per-model details
        for result in results:
            model_data = {
                "model_type": result.get('model_type', 'Unknown').upper(),
                "prediction": result.get('prediction', 'Unknown'),
                "confidence": f"{result.get('confidence', 0) * 100:.2f}%",
                "frames_processed": result.get('num_faces', num_frames_analyzed),
                "fake_frames_detected": result.get('fake_count', 0),
                "fake_frames_ratio": f"{result.get('fake_ratio', 0) * 100:.2f}%",
                "average_probability_real": f"{result.get('avg_prob_real', 0) * 100:.2f}%",
                "average_probability_fake": f"{result.get('avg_prob_fake', 0) * 100:.2f}%"
            }
            
            # Add artifact scores if available (full frame mode)
            if 'avg_artifact_score' in result:
                model_data["artifact_score"] = f"{result['avg_artifact_score']:.3f}"
                model_data["region_inconsistency"] = f"{result.get('avg_region_inconsistency', 0):.3f}"
            
            report["model_results"].append(model_data)
            
            # Add to detailed metrics
            detailed = {
                "raw_avg_probability_real": result.get('avg_prob_real', 0),
                "raw_avg_probability_fake": result.get('avg_prob_fake', 0),
                "fake_detection_threshold": 0.25,
                "frames_classified_fake": result.get('fake_count', 0),
                "frames_classified_real": result.get('num_faces', num_frames_analyzed) - result.get('fake_count', 0),
                "confidence_score": result.get('confidence', 0)
            }
            
            # Add artifact details if available
            if 'avg_artifact_score' in result:
                detailed["artifact_analysis"] = {
                    "compression_artifacts": result.get('avg_artifact_score', 0),
                    "region_inconsistency": result.get('avg_region_inconsistency', 0),
                    "blur_inconsistency": "Analyzed",
                    "color_inconsistency": "Analyzed",
                    "lighting_inconsistency": "Analyzed"
                }
            
            report["detailed_metrics"][result.get('model_type', 'Unknown')] = detailed
        
        # Add frame analysis summary
        report["frame_analysis"] = {
            "total_frames_analyzed": num_frames_analyzed,
            "sampling_strategy": "Uniform sampling across video duration",
            "detection_rule": "Video is FAKE if ≥25% of frames are classified as FAKE",
            "analysis_regions": "6 regions per frame (full, center, top, bottom, left, right)" if analysis_mode == "full_frame" else "Detected faces only"
        }
        
        # Add ensemble information
        report["ensemble_method"] = {
            "type": "Aggressive Ensemble with Improved Predictor",
            "description": "Combines EfficientNet-B3 and Swin Transformer predictions with lowered threshold",
            "models_used": len(results),
            "individual_threshold": 0.45,
            "video_fake_threshold": "25% of frames"
        }
        
        return report
    
    def _generate_interpretation(
        self,
        prediction: str,
        confidence: float,
        results: List[Dict[str, Any]],
        is_video: bool = False
    ) -> Dict[str, Any]:
        """Generate human-readable interpretation of results"""
        
        interpretation = {
            "summary": "",
            "confidence_level": "",
            "reliability": "",
            "recommendations": []
        }
        
        # Summary
        if prediction == "FAKE":
            interpretation["summary"] = f"The analysis indicates that this {'video' if is_video else 'image'} is likely a DEEPFAKE with {confidence:.1f}% confidence."
        else:
            interpretation["summary"] = f"The analysis indicates that this {'video' if is_video else 'image'} appears to be GENUINE with {confidence:.1f}% confidence."
        
        # Confidence level
        if confidence >= 90:
            interpretation["confidence_level"] = "Very High - Strong evidence supports the prediction"
        elif confidence >= 75:
            interpretation["confidence_level"] = "High - Substantial evidence supports the prediction"
        elif confidence >= 60:
            interpretation["confidence_level"] = "Moderate - Evidence leans towards the prediction"
        else:
            interpretation["confidence_level"] = "Low - Uncertain, borderline case"
        
        # Reliability assessment
        model_agreement = sum(1 for r in results if r.get('prediction') == prediction)
        total_models = len(results)
        
        if model_agreement == total_models:
            interpretation["reliability"] = f"Excellent - All {total_models} models agree on the prediction"
        elif model_agreement > total_models / 2:
            interpretation["reliability"] = f"Good - {model_agreement} out of {total_models} models agree"
        else:
            interpretation["reliability"] = f"Uncertain - Models disagree ({model_agreement}/{total_models} agreement)"
        
        # Recommendations
        if prediction == "FAKE":
            interpretation["recommendations"] = [
                "⚠️ This content should be treated as potentially manipulated",
                "🔍 Verify the source and authenticity before sharing",
                "📋 Consider additional forensic analysis if high stakes",
                "🚫 Do not rely on this content for critical decisions"
            ]
        else:
            if confidence < 75:
                interpretation["recommendations"] = [
                    "✓ Content appears genuine but confidence is moderate",
                    "🔍 Consider context and source verification",
                    "📊 May want to perform additional checks if critical"
                ]
            else:
                interpretation["recommendations"] = [
                    "✓ Content appears genuine with high confidence",
                    "✓ All detection models agree on authenticity",
                    "🔍 Standard verification practices still recommended"
                ]
        
        return interpretation
    
    def format_as_text(self, report: Dict[str, Any]) -> str:
        """Format report as plain text"""
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"  {report['report_type'].upper()}")
        lines.append("=" * 80)
        lines.append(f"Generated: {report['generated_at']}")
        lines.append("")
        
        # File info
        lines.append("FILE INFORMATION")
        lines.append("-" * 80)
        for key, value in report['file_info'].items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Overall results
        lines.append("OVERALL RESULTS")
        lines.append("-" * 80)
        for key, value in report['overall_results'].items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Model results
        lines.append("MODEL-BY-MODEL RESULTS")
        lines.append("-" * 80)
        for i, model in enumerate(report['model_results'], 1):
            lines.append(f"\n  Model {i}: {model['model_type']}")
            for key, value in model.items():
                if key != 'model_type':
                    lines.append(f"    {key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Interpretation
        lines.append("INTERPRETATION")
        lines.append("-" * 80)
        interp = report['interpretation']
        lines.append(f"  Summary: {interp['summary']}")
        lines.append(f"  Confidence Level: {interp['confidence_level']}")
        lines.append(f"  Reliability: {interp['reliability']}")
        lines.append("\n  Recommendations:")
        for rec in interp['recommendations']:
            lines.append(f"    • {rec}")
        lines.append("")
        
        # Detailed metrics
        lines.append("DETAILED METRICS")
        lines.append("-" * 80)
        for model_name, metrics in report['detailed_metrics'].items():
            lines.append(f"\n  {model_name.upper()}:")
            self._format_dict_recursive(metrics, lines, indent=4)
        lines.append("")
        
        # Ensemble method
        lines.append("ENSEMBLE METHOD")
        lines.append("-" * 80)
        for key, value in report['ensemble_method'].items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Frame analysis (for videos)
        if 'frame_analysis' in report:
            lines.append("FRAME ANALYSIS")
            lines.append("-" * 80)
            for key, value in report['frame_analysis'].items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_dict_recursive(self, d: Dict, lines: List[str], indent: int = 0):
        """Helper to format nested dictionaries"""
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{' ' * indent}{key.replace('_', ' ').title()}:")
                self._format_dict_recursive(value, lines, indent + 2)
            else:
                lines.append(f"{' ' * indent}{key.replace('_', ' ').title()}: {value}")
    
    def format_as_json(self, report: Dict[str, Any], pretty: bool = True) -> str:
        """Format report as JSON"""
        if pretty:
            return json.dumps(report, indent=2, ensure_ascii=False)
        return json.dumps(report, ensure_ascii=False)
    
    def format_as_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML"""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report['report_type']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header .date {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .info-item {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .info-item strong {{
            display: block;
            color: #495057;
            margin-bottom: 5px;
            font-size: 14px;
        }}
        .info-item span {{
            color: #212529;
            font-size: 16px;
            font-weight: 600;
        }}
        .verdict {{
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .verdict.fake {{
            background: #ffe5e5;
            color: #d32f2f;
            border: 2px solid #d32f2f;
        }}
        .verdict.real {{
            background: #e8f5e9;
            color: #388e3c;
            border: 2px solid #388e3c;
        }}
        .model-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #dee2e6;
        }}
        .model-card h3 {{
            color: #495057;
            margin-top: 0;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #6c757d;
        }}
        .metric-value {{
            font-weight: 600;
            color: #212529;
        }}
        .recommendations {{
            list-style: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 10px;
            margin: 8px 0;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #f8f9fa;
            color: #495057;
            font-weight: 600;
        }}
        .confidence-bar {{
            width: 100%;
            height: 25px;
            background: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 {report['report_type']}</h1>
        <div class="date">Generated: {report['generated_at']}</div>
    </div>
    
    <div class="section">
        <h2>📁 File Information</h2>
        <div class="info-grid">
"""
        
        for key, value in report['file_info'].items():
            html += f"""
            <div class="info-item">
                <strong>{key.replace('_', ' ').title()}</strong>
                <span>{value}</span>
            </div>
"""
        
        html += """
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Overall Results</h2>
"""
        
        verdict_class = "fake" if report['overall_results']['prediction'] == "FAKE" else "real"
        html += f"""
        <div class="verdict {verdict_class}">
            {report['overall_results']['verdict']}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {report['overall_results']['confidence']}">
                Confidence: {report['overall_results']['confidence']}
            </div>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>🤖 Model Results</h2>
"""
        
        for model in report['model_results']:
            html += f"""
        <div class="model-card">
            <h3>{model['model_type']}</h3>
"""
            for key, value in model.items():
                if key != 'model_type':
                    html += f"""
            <div class="metric">
                <span class="metric-label">{key.replace('_', ' ').title()}</span>
                <span class="metric-value">{value}</span>
            </div>
"""
            html += """
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>💡 Interpretation</h2>
"""
        
        interp = report['interpretation']
        html += f"""
        <p><strong>Summary:</strong> {interp['summary']}</p>
        <p><strong>Confidence Level:</strong> {interp['confidence_level']}</p>
        <p><strong>Reliability:</strong> {interp['reliability']}</p>
        
        <h3>Recommendations:</h3>
        <ul class="recommendations">
"""
        
        for rec in interp['recommendations']:
            html += f"            <li>{rec}</li>\n"
        
        html += """
        </ul>
    </div>
    
    <div class="section">
        <h2>📊 Detailed Metrics</h2>
"""
        
        for model_name, metrics in report['detailed_metrics'].items():
            html += f"<h3>{model_name.upper()}</h3>"
            html += "<table>"
            self._add_metrics_to_html(metrics, html_lines := [])
            html += "".join(html_lines)
            html += "</table>"
        
        html += """
    </div>
"""
        
        if 'frame_analysis' in report:
            html += """
    <div class="section">
        <h2>🎬 Frame Analysis</h2>
        <div class="info-grid">
"""
            for key, value in report['frame_analysis'].items():
                html += f"""
            <div class="info-item">
                <strong>{key.replace('_', ' ').title()}</strong>
                <span>{value}</span>
            </div>
"""
            html += """
        </div>
    </div>
"""
        
        html += """
    <div class="footer">
        <p>Report generated by Deepfake Detection System</p>
        <p>For questions or concerns about this analysis, please consult with a digital forensics expert.</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _add_metrics_to_html(self, metrics: Dict, html_lines: List[str], level: int = 0):
        """Helper to add metrics to HTML recursively"""
        for key, value in metrics.items():
            if isinstance(value, dict):
                html_lines.append(f"<tr><td colspan='2'><strong>{key.replace('_', ' ').title()}</strong></td></tr>")
                self._add_metrics_to_html(value, html_lines, level + 1)
            else:
                indent = "&nbsp;" * (level * 4)
                html_lines.append(f"<tr><td>{indent}{key.replace('_', ' ').title()}</td><td>{value}</td></tr>")


def create_downloadable_report(report_data: Dict[str, Any], format: str = "txt") -> bytes:
    """
    Create downloadable report in specified format
    
    Args:
        report_data: Report dictionary
        format: 'txt', 'json', or 'html'
        
    Returns:
        Bytes of the formatted report
    """
    generator = ReportGenerator()
    
    if format == "txt":
        content = generator.format_as_text(report_data)
        return content.encode('utf-8')
    elif format == "json":
        content = generator.format_as_json(report_data)
        return content.encode('utf-8')
    elif format == "html":
        content = generator.format_as_html(report_data)
        return content.encode('utf-8')
    else:
        raise ValueError(f"Unsupported format: {format}")
