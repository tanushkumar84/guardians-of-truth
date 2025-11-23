"""
Comprehensive Evaluation and Error Analysis Tools
Provides detailed model evaluation, error analysis, and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    average_precision_score, precision_recall_curve, roc_curve
)
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import json
from collections import defaultdict


class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate(self, dataloader: DataLoader, threshold: float = 0.5) -> Dict:
        """
        Comprehensive evaluation with detailed metrics
        
        Returns:
            Dictionary containing all metrics and predictions
        """
        all_preds = []
        all_probs = []
        all_labels = []
        all_losses = []
        
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs.squeeze(), labels)
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs > threshold).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_losses.append(loss.item())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['avg_loss'] = np.mean(all_losses)
        
        # Store for error analysis
        self.predictions = {
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'metrics': metrics
        }
        
        return metrics
    
    def _calculate_metrics(self, labels: np.ndarray, preds: np.ndarray, 
                          probs: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        try:
            auc_roc = roc_auc_score(labels, probs)
            auc_pr = average_precision_score(labels, probs)
        except:
            auc_roc = 0.0
            auc_pr = 0.0
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # False positive/negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'false_positive_rate': fpr,
            'false_negative_rate': fnr
        }
    
    def print_report(self, save_path: Optional[str] = None):
        """Print and optionally save detailed evaluation report"""
        
        metrics = self.predictions['metrics']
        
        report = f"""
{'='*60}
MODEL EVALUATION REPORT
{'='*60}

OVERALL PERFORMANCE:
  Accuracy:          {metrics['accuracy']:.4f}
  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}
  F1 Score:          {metrics['f1_score']:.4f}
  AUC-ROC:           {metrics['auc_roc']:.4f}
  AUC-PR:            {metrics['auc_pr']:.4f}

DEEPFAKE DETECTION (Class 1):
  Precision:         {metrics['precision']:.4f}
  Recall:            {metrics['recall']:.4f}
  True Positives:    {metrics['true_positives']}
  False Positives:   {metrics['false_positives']}
  False Negatives:   {metrics['false_negatives']}

REAL IMAGE DETECTION (Class 0):
  Specificity:       {metrics['specificity']:.4f}
  True Negatives:    {metrics['true_negatives']}

ERROR RATES:
  False Positive Rate: {metrics['false_positive_rate']:.4f}
  False Negative Rate: {metrics['false_negative_rate']:.4f}

{'='*60}
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            
            # Save metrics as JSON
            json_path = save_path.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)


class ErrorAnalyzer:
    """Analyze model errors and failure cases"""
    
    def __init__(self, predictions: Dict):
        self.preds = predictions['predictions']
        self.probs = predictions['probabilities']
        self.labels = predictions['labels']
        
    def get_error_indices(self) -> Dict[str, np.ndarray]:
        """Get indices of different error types"""
        
        errors = {
            'false_positives': np.where((self.preds == 1) & (self.labels == 0))[0],
            'false_negatives': np.where((self.preds == 0) & (self.labels == 1))[0],
            'true_positives': np.where((self.preds == 1) & (self.labels == 1))[0],
            'true_negatives': np.where((self.preds == 0) & (self.labels == 0))[0]
        }
        
        return errors
    
    def analyze_confidence(self) -> Dict:
        """Analyze prediction confidence for different cases"""
        
        errors = self.get_error_indices()
        
        analysis = {}
        for error_type, indices in errors.items():
            if len(indices) > 0:
                confidences = self.probs[indices]
                analysis[error_type] = {
                    'count': len(indices),
                    'mean_confidence': float(np.mean(confidences)),
                    'std_confidence': float(np.std(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences)),
                    'median_confidence': float(np.median(confidences))
                }
        
        return analysis
    
    def get_high_confidence_errors(self, confidence_threshold: float = 0.9) -> Dict:
        """Find errors where model was very confident but wrong"""
        
        errors = self.get_error_indices()
        
        high_conf_errors = {}
        for error_type in ['false_positives', 'false_negatives']:
            indices = errors[error_type]
            if len(indices) > 0:
                confidences = self.probs[indices]
                
                if error_type == 'false_positives':
                    high_conf = indices[confidences > confidence_threshold]
                else:  # false_negatives
                    high_conf = indices[confidences < (1 - confidence_threshold)]
                
                high_conf_errors[error_type] = high_conf
        
        return high_conf_errors
    
    def print_error_analysis(self):
        """Print detailed error analysis"""
        
        conf_analysis = self.analyze_confidence()
        high_conf_errors = self.get_high_confidence_errors()
        
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        for error_type, stats in conf_analysis.items():
            print(f"\n{error_type.upper().replace('_', ' ')}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean Confidence: {stats['mean_confidence']:.4f} ± {stats['std_confidence']:.4f}")
            print(f"  Range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
            print(f"  Median: {stats['median_confidence']:.4f}")
        
        print("\n" + "-"*60)
        print("HIGH CONFIDENCE ERRORS (>0.9):")
        for error_type, indices in high_conf_errors.items():
            print(f"  {error_type}: {len(indices)} cases")


class VisualizationTools:
    """Advanced visualization tools for model evaluation"""
    
    @staticmethod
    def plot_comprehensive_evaluation(predictions: Dict, save_dir: str):
        """Create comprehensive evaluation visualization"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        preds = predictions['predictions']
        probs = predictions['probabilities']
        labels = predictions['labels']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_roc = roc_auc_score(labels, probs)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC={auc_roc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        auc_pr = average_precision_score(labels, probs)
        axes[0, 2].plot(recall, precision, label=f'PR (AUC={auc_pr:.4f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confidence Distribution
        axes[1, 0].hist(probs[labels == 0], bins=50, alpha=0.5, label='Real', color='blue')
        axes[1, 0].hist(probs[labels == 1], bins=50, alpha=0.5, label='Fake', color='red')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].legend()
        axes[1, 0].axvline(0.5, color='black', linestyle='--', alpha=0.5)
        
        # 5. Error Distribution
        errors = (preds != labels).astype(int)
        error_probs = probs[errors == 1]
        axes[1, 1].hist(error_probs, bins=30, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Number of Errors')
        axes[1, 1].set_title('Error Confidence Distribution')
        axes[1, 1].axvline(0.5, color='red', linestyle='--', alpha=0.5)
        
        # 6. Calibration Plot
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_accs = []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if mask.sum() > 0:
                bin_accs.append(labels[mask].mean())
            else:
                bin_accs.append(0)
        
        axes[1, 2].plot(bin_centers, bin_accs, 'o-', label='Model')
        axes[1, 2].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Actual Frequency')
        axes[1, 2].set_title('Calibration Plot')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comprehensive_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_class_distribution(labels: np.ndarray, save_path: str):
        """Plot class distribution"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        unique, counts = np.unique(labels, return_counts=True)
        colors = ['blue', 'red']
        labels_text = ['Real', 'Fake']
        
        ax.bar(labels_text, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        
        # Add count labels
        for i, (label, count) in enumerate(zip(labels_text, counts)):
            ax.text(i, count, f'{count}\n({count/len(labels)*100:.1f}%)', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_model_comprehensive(model: nn.Module, test_loader: DataLoader, 
                                 device: torch.device, save_dir: str) -> Dict:
    """
    Run comprehensive model evaluation
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all evaluation results
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    # Print report
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    evaluator.print_report(report_path)
    
    # Error analysis
    print("\nPerforming error analysis...")
    analyzer = ErrorAnalyzer(evaluator.predictions)
    analyzer.print_error_analysis()
    
    # Visualizations
    print("\nCreating visualizations...")
    VisualizationTools.plot_comprehensive_evaluation(
        evaluator.predictions, save_dir
    )
    
    # Extract labels for class distribution
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    VisualizationTools.plot_class_distribution(
        all_labels, 
        os.path.join(save_dir, 'class_distribution.png')
    )
    
    print(f"\nEvaluation complete! Results saved to {save_dir}")
    
    return {
        'metrics': metrics,
        'predictions': evaluator.predictions,
        'error_analysis': analyzer.analyze_confidence()
    }


if __name__ == "__main__":
    # Test example
    print("Evaluation Tools Module")
    print("="*60)
    print("\nThis module provides comprehensive model evaluation tools:")
    print("  - ModelEvaluator: Detailed metrics calculation")
    print("  - ErrorAnalyzer: Error pattern analysis")
    print("  - VisualizationTools: Advanced visualizations")
    print("\nUsage:")
    print("  from evaluation_tools import evaluate_model_comprehensive")
    print("  results = evaluate_model_comprehensive(model, test_loader, device, './results')")
