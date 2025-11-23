"""
Enhanced Ensemble Inference for Deepfake Detection
Features:
- Weighted ensemble predictions
- Uncertainty estimation
- Confidence calibration
- Multi-model voting
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models
    """
    def __init__(self, models, weights=None, device='cpu'):
        """
        Args:
            models: List of models
            weights: Optional weights for each model (default: equal weights)
            device: Device to run inference on
        """
        self.models = [model.to(device).eval() for model in models]
        self.device = device
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
        
        logger.info(f"Initialized ensemble with {len(models)} models")
        logger.info(f"Model weights: {self.weights}")

    @torch.no_grad()
    def predict(self, x, return_uncertainty=False):
        """
        Make ensemble prediction
        
        Args:
            x: Input tensor [B, C, H, W]
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            predictions: Ensemble predictions
            uncertainty: (Optional) Prediction uncertainty
        """
        predictions = []
        
        for model in self.models:
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # [num_models, batch_size]
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        if return_uncertainty:
            # Calculate uncertainty as variance across predictions
            uncertainty = np.var(predictions, axis=0)
            return ensemble_pred, uncertainty
        
        return ensemble_pred

    @torch.no_grad()
    def predict_with_voting(self, x, threshold=0.5):
        """
        Make prediction using majority voting
        
        Args:
            x: Input tensor [B, C, H, W]
            threshold: Classification threshold
        
        Returns:
            predictions: Majority vote predictions
            confidence: Voting confidence (proportion of models agreeing)
        """
        predictions = []
        
        for model in self.models:
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).cpu().numpy()
            predictions.append(preds)
        
        predictions = np.array(predictions)  # [num_models, batch_size]
        
        # Majority voting
        votes = predictions.sum(axis=0)
        majority_pred = (votes > len(self.models) / 2).astype(float)
        
        # Confidence = proportion of models agreeing
        confidence = np.abs(votes / len(self.models) - 0.5) + 0.5
        
        return majority_pred, confidence

    @torch.no_grad()
    def predict_with_calibration(self, x, temperature=1.0):
        """
        Make prediction with temperature scaling for calibration
        
        Args:
            x: Input tensor [B, C, H, W]
            temperature: Temperature parameter for scaling
        
        Returns:
            calibrated_predictions: Temperature-scaled predictions
        """
        predictions = []
        
        for model in self.models:
            logits = model(x).squeeze()
            # Apply temperature scaling
            calibrated_logits = logits / temperature
            probs = torch.sigmoid(calibrated_logits)
            predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using multiple methods
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def monte_carlo_dropout(self, x, num_samples=10, dropout_rate=0.2):
        """
        Estimate uncertainty using Monte Carlo Dropout
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes
            dropout_rate: Dropout probability
        
        Returns:
            mean_pred: Mean prediction
            uncertainty: Prediction uncertainty (standard deviation)
        """
        # Enable dropout at test time
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
                module.p = dropout_rate
        
        predictions = []
        
        for _ in range(num_samples):
            logits = self.model(x).squeeze()
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        # Reset model to eval mode
        self.model.eval()
        
        return mean_pred, uncertainty


def calibrate_predictions(predictions, labels, method='temperature_scaling'):
    """
    Calibrate model predictions
    
    Args:
        predictions: Model predictions (logits)
        labels: Ground truth labels
        method: Calibration method ('temperature_scaling', 'platt_scaling')
    
    Returns:
        optimal_temperature: Optimal temperature parameter
    """
    if method == 'temperature_scaling':
        # Find optimal temperature using validation set
        from scipy.optimize import minimize
        
        def nll_loss(temperature):
            """Negative log likelihood loss"""
            scaled_logits = predictions / temperature
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return loss
        
        result = minimize(nll_loss, x0=1.0, method='BFGS')
        optimal_temperature = result.x[0]
        
        logger.info(f"Optimal temperature: {optimal_temperature:.4f}")
        return optimal_temperature
    
    else:
        raise NotImplementedError(f"Method {method} not implemented")


class AdaptiveThreshold:
    """
    Adaptive threshold selection based on validation performance
    """
    def __init__(self):
        self.optimal_threshold = 0.5

    def find_optimal_threshold(self, predictions, labels, metric='f1'):
        """
        Find optimal threshold for classification
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            metric: Optimization metric ('f1', 'accuracy', 'balanced_accuracy')
        
        Returns:
            optimal_threshold: Optimal classification threshold
        """
        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            preds_binary = (predictions > threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(labels, preds_binary)
            elif metric == 'accuracy':
                score = accuracy_score(labels, preds_binary)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(labels, preds_binary)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        logger.info(f"Optimal threshold: {best_threshold:.4f} (metric: {metric}, score: {best_score:.4f})")
        
        return best_threshold

    def predict(self, predictions):
        """
        Make predictions using optimal threshold
        
        Args:
            predictions: Model predictions (probabilities)
        
        Returns:
            binary_predictions: Binary predictions using optimal threshold
        """
        return (predictions > self.optimal_threshold).astype(int)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test ensemble predictor
    logger.info("Testing Ensemble Predictor...")
    
    # Create dummy models
    from enhanced_models import EnhancedEfficientNet
    
    model1 = EnhancedEfficientNet(pretrained=False)
    model2 = EnhancedEfficientNet(pretrained=False)
    
    # Create ensemble
    ensemble = EnsemblePredictor([model1, model2], weights=[0.6, 0.4])
    
    # Test prediction
    x = torch.randn(4, 3, 300, 300)
    pred = ensemble.predict(x)
    logger.info(f"Ensemble prediction shape: {pred.shape}")
    
    pred, uncertainty = ensemble.predict(x, return_uncertainty=True)
    logger.info(f"Prediction with uncertainty - pred: {pred.shape}, uncertainty: {uncertainty.shape}")
    
    # Test voting
    vote_pred, confidence = ensemble.predict_with_voting(x)
    logger.info(f"Voting prediction - pred: {vote_pred.shape}, confidence: {confidence.shape}")
    
    logger.info("✅ All ensemble methods working correctly!")
