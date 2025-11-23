"""
Advanced Training Pipeline for Deepfake Detection
Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Advanced learning rate schedulers
- Early stopping
- Model checkpointing
- Comprehensive metrics tracking
- Test-time augmentation
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from tqdm import tqdm
from pathlib import Path
import logging
from collections import defaultdict
import json

from enhanced_models import (
    EnhancedEfficientNet, 
    EnhancedSwinTransformer,
    get_loss_function
)
from advanced_augmentation import MixupCutmix, TestTimeAugmentation
from data_handler import get_dataloaders

logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Advanced trainer with all bells and whistles
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = get_loss_function(
            config.get('loss_type', 'focal'),
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer with differential learning rates
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Mixup/Cutmix
        self.mixup_cutmix = MixupCutmix(
            mixup_alpha=config.get('mixup_alpha', 1.0),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            prob=config.get('mixup_prob', 0.5)
        ) if config.get('use_mixup_cutmix', True) else None
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Early stopping
        self.patience = config.get('patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics storage
        self.history = defaultdict(list)
        
        # Model save directory
        self.save_dir = Path(config.get('save_dir', f'models/{config["model_type"]}'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized AdvancedTrainer with config: {config}")

    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates"""
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {
                'params': backbone_params,
                'lr': self.config.get('backbone_lr', 1e-5),
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            {
                'params': classifier_params,
                'lr': self.config.get('classifier_lr', 1e-4),
                'weight_decay': self.config.get('weight_decay', 0.01)
            }
        ])
        
        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-7)
            )
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('scheduler_patience', 3),
                factor=self.config.get('scheduler_factor', 0.1)
            )
        elif scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[
                    self.config.get('backbone_lr', 1e-5) * 10,
                    self.config.get('classifier_lr', 1e-4) * 10
                ],
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        else:
            scheduler = None
        
        return scheduler

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets_list = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply Mixup/Cutmix
            if self.mixup_cutmix is not None and epoch > 5:  # Start after warmup
                inputs, labels = self.mixup_cutmix(inputs, labels)
            
            # Mixed precision training
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update scheduler (for OneCycleLR)
                if self.config.get('scheduler') == 'onecycle':
                    self.scheduler.step()
            
            # Track metrics
            running_loss += loss.item() * self.accumulation_steps
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                predictions.extend(probs.cpu().numpy())
                targets_list.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item() * self.accumulation_steps})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        predictions = np.array(predictions)
        targets_arr = np.array(targets_list)
        
        # Accuracy
        accuracy = ((predictions > 0.5) == targets_arr).mean()
        
        # F1 Score
        f1 = f1_score(targets_arr, predictions > 0.5)
        
        return epoch_loss, accuracy, f1, predictions, targets_arr

    @torch.no_grad()
    def validate(self, loader, use_tta=False):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets_list = []
        
        # Test-Time Augmentation
        if use_tta:
            tta = TestTimeAugmentation(self.model, num_augments=5)
        
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if use_tta:
                outputs = tta(inputs).squeeze()
            else:
                outputs = self.model(inputs).squeeze()
            
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
            targets_list.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(loader)
        predictions = np.array(predictions)
        targets_arr = np.array(targets_list)
        
        accuracy = ((predictions > 0.5) == targets_arr).mean()
        f1 = f1_score(targets_arr, predictions > 0.5)
        
        return avg_loss, accuracy, f1, predictions, targets_arr

    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log metrics to MLflow"""
        train_loss, train_acc, train_f1 = train_metrics
        val_loss, val_acc, val_f1 = val_metrics
        
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        # Store in history
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_acc)
        self.history['train_f1'].append(train_f1)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
        self.history['val_f1'].append(val_f1)

    def plot_confusion_matrix(self, y_true, y_pred, phase='train', epoch=None):
        """Plot and log confusion matrix"""
        cm = confusion_matrix(y_true, y_pred > 0.5)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{phase} Confusion Matrix (Epoch {epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        mlflow.log_figure(plt.gcf(), f'{phase}_confusion_matrix_epoch{epoch}.png')
        plt.close()

    def plot_roc_curve(self, y_true, y_pred, phase='train', epoch=None):
        """Plot and log ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{phase} ROC Curve (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        mlflow.log_figure(plt.gcf(), f'{phase}_roc_curve_epoch{epoch}.png')
        plt.close()
        
        return roc_auc

    def plot_precision_recall_curve(self, y_true, y_pred, phase='train', epoch=None):
        """Plot and log Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{phase} Precision-Recall Curve (Epoch {epoch})')
        plt.grid(True, alpha=0.3)
        
        mlflow.log_figure(plt.gcf(), f'{phase}_pr_curve_epoch{epoch}.png')
        plt.close()

    def plot_learning_curves(self):
        """Plot comprehensive learning curves"""
        epochs = range(len(self.history['train_loss']))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score over Epochs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view
        ax_twin = axes[1, 1].twinx()
        axes[1, 1].plot(epochs, self.history['val_loss'], 'b-', label='Val Loss')
        ax_twin.plot(epochs, self.history['val_accuracy'], 'r-', label='Val Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='b')
        ax_twin.set_ylabel('Accuracy', color='r')
        axes[1, 1].set_title('Validation Loss & Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(fig, 'learning_curves.png')
        plt.close()

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch{epoch}_acc{val_acc:.4f}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model_cpu.pth'
            torch.save(self.model.state_dict(), best_path, _use_new_zipfile_serialization=False)
            logger.info(f"✅ Saved best model with accuracy: {val_acc:.4f}")
            
            # Log to MLflow
            mlflow.pytorch.log_model(self.model, "best_model")

    def train(self):
        """Main training loop"""
        num_epochs = self.config['num_epochs']
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss, train_acc, train_f1, train_preds, train_targets = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_targets = self.validate(self.val_loader)
            
            # Log metrics
            self.log_metrics(epoch, (train_loss, train_acc, train_f1), (val_loss, val_acc, val_f1))
            
            # Plot metrics every 5 epochs
            if epoch % 5 == 0:
                self.plot_confusion_matrix(train_targets, train_preds, 'train', epoch)
                self.plot_confusion_matrix(val_targets, val_preds, 'val', epoch)
                self.plot_roc_curve(train_targets, train_preds, 'train', epoch)
                roc_auc = self.plot_roc_curve(val_targets, val_preds, 'val', epoch)
                self.plot_precision_recall_curve(val_targets, val_preds, 'val', epoch)
                
                mlflow.log_metric('val_roc_auc', roc_auc, step=epoch)
            
            # Learning rate scheduling
            if self.config.get('scheduler') == 'plateau':
                self.scheduler.step(val_loss)
            elif self.config.get('scheduler') == 'cosine':
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Log progress
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Plot final learning curves
        self.plot_learning_curves()
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        mlflow.log_artifact(str(history_path))

    def test(self, use_tta=True):
        """Test the model with optional TTA"""
        logger.info("\n" + "="*50)
        logger.info("Testing with best model...")
        logger.info("="*50)
        
        # Load best model
        best_model_path = self.save_dir / 'best_model_cpu.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Test
        test_loss, test_acc, test_f1, test_preds, test_targets = self.validate(
            self.test_loader,
            use_tta=use_tta
        )
        
        # Log test metrics
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': test_f1
        })
        
        # Plot test results
        self.plot_confusion_matrix(test_targets, test_preds, 'test')
        test_roc_auc = self.plot_roc_curve(test_targets, test_preds, 'test')
        self.plot_precision_recall_curve(test_targets, test_preds, 'test')
        
        mlflow.log_metric('test_roc_auc', test_roc_auc)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")
        logger.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
        
        return test_loss, test_acc, test_f1, test_roc_auc


def main():
    # Configuration
    config = {
        # Model
        'model_type': 'efficientnet',  # or 'swin'
        'model_name': 'efficientnet-b3',  # or 'swin_base_patch4_window7_224'
        
        # Data
        'data_dir': '/kaggle/input/2body-images-10k-split',
        'image_size': 300,  # 300 for EfficientNet, 224 for Swin
        'batch_size': 16,  # Reduced for larger models + AMP
        
        # Training
        'num_epochs': 30,
        'backbone_lr': 1e-5,
        'classifier_lr': 1e-4,
        'weight_decay': 0.01,
        
        # Loss
        'loss_type': 'focal',  # 'bce', 'focal', 'label_smoothing'
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        
        # Augmentation
        'use_mixup_cutmix': True,
        'mixup_alpha': 1.0,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,
        
        # Advanced training
        'use_amp': True,  # Mixed precision
        'accumulation_steps': 2,  # Gradient accumulation
        
        # Scheduler
        'scheduler': 'cosine',  # 'cosine', 'plateau', 'onecycle'
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-7,
        
        # Early stopping
        'patience': 10,
        
        # Save
        'save_dir': 'models/efficientnet_enhanced'
    }
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MLflow
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment(f'deepfake_{config["model_type"]}_enhanced')
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        config['data_dir'],
        config['image_size'],
        config['batch_size']
    )
    
    # Model
    if config['model_type'] == 'efficientnet':
        model = EnhancedEfficientNet(model_name=config['model_name'])
    else:
        model = EnhancedSwinTransformer(model_name=config['model_name'])
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params(config)
        
        # Train
        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=config
        )
        
        trainer.train()
        trainer.test(use_tta=True)
        
        logger.info("\n✅ Training completed successfully!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
