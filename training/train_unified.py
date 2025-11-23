"""
Unified Advanced Training Script
Combines all improvements: advanced augmentation, enhanced models, training techniques, and evaluation
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from pathlib import Path

# Import custom modules
from data_handler import DeepfakeDataset
from advanced_augmentation import get_advanced_transforms
from enhanced_models import get_enhanced_model, get_loss_function
from train_advanced import AdvancedTrainer
from ensemble_inference import EnsemblePredictor
from evaluation_tools import evaluate_model_comprehensive


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Deepfake Detection Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers (default: 4)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='efficientnet',
                      choices=['efficientnet', 'swin'],
                      help='Model architecture (default: efficientnet)')
    parser.add_argument('--use_attention', action='store_true',
                      help='Use attention mechanisms in model')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate (default: 0.3)')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size (default: 224)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=30,
                      help='Number of training epochs (default: 30)')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                      help='Learning rate for backbone (default: 1e-5)')
    parser.add_argument('--classifier_lr', type=float, default=1e-4,
                      help='Learning rate for classifier (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['cosine', 'plateau', 'onecycle'],
                      help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                      help='Number of warmup epochs (default: 3)')
    
    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default='focal',
                      choices=['bce', 'focal', 'label_smoothing'],
                      help='Loss function type (default: focal)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                      help='Focal loss alpha (default: 0.25)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help='Focal loss gamma (default: 2.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                      help='Label smoothing factor (default: 0.1)')
    
    # Augmentation arguments
    parser.add_argument('--use_advanced_aug', action='store_true',
                      help='Use advanced augmentations (RandAugment, Mixup, etc.)')
    parser.add_argument('--use_mixup', action='store_true',
                      help='Use Mixup/CutMix augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                      help='Mixup alpha parameter (default: 0.2)')
    parser.add_argument('--use_tta', action='store_true',
                      help='Use test-time augmentation for evaluation')
    
    # Advanced training arguments
    parser.add_argument('--use_amp', action='store_true',
                      help='Use automatic mixed precision training')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                      help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Early stopping patience (default: 10)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                      help='Gradient clipping value (default: 1.0)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./runs',
                      help='Output directory for models and logs (default: ./runs)')
    parser.add_argument('--experiment_name', type=str, default='deepfake_detection',
                      help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                      help='MLflow run name (default: auto-generated)')
    
    # Evaluation arguments
    parser.add_argument('--eval_only', action='store_true',
                      help='Only run evaluation (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint for evaluation')
    
    return parser.parse_args()


def setup_data_loaders(args):
    """Setup train/val/test data loaders with advanced augmentations"""
    
    print(f"Loading data from {args.data_dir}")
    
    # Get transforms
    if args.use_advanced_aug:
        train_transform, val_transform = get_advanced_transforms(
            image_size=args.image_size,
            use_randaugment=True,
            use_deepfake_aug=True
        )
        print("Using advanced augmentations (RandAugment + Deepfake-specific)")
    else:
        from data_handler import get_transforms
        train_transform, val_transform = get_transforms(args.image_size)
        print("Using basic augmentations")
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = DeepfakeDataset(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    test_dataset = DeepfakeDataset(
        os.path.join(args.data_dir, 'test'),
        transform=val_transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def setup_model_and_loss(args, device):
    """Setup model and loss function"""
    
    print(f"\nInitializing {args.model_type.upper()} model...")
    
    # Create model
    model = get_enhanced_model(
        model_type=args.model_type,
        pretrained=True,
        num_classes=1,
        use_attention=args.use_attention,
        dropout=args.dropout
    )
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function
    criterion = get_loss_function(
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing
    )
    
    print(f"Using {args.loss_type} loss")
    
    return model, criterion


def train_model(args, model, criterion, train_loader, val_loader, device):
    """Train the model with advanced training techniques"""
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model_type}")
    print(f"Attention: {args.use_attention}")
    print(f"Loss: {args.loss_type}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Backbone LR: {args.backbone_lr}")
    print(f"Classifier LR: {args.classifier_lr}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Mixed Precision: {args.use_amp}")
    print(f"Gradient Accumulation: {args.gradient_accumulation}")
    print(f"Advanced Augmentation: {args.use_advanced_aug}")
    print(f"Mixup/CutMix: {args.use_mixup}")
    print("="*60 + "\n")
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        criterion=criterion,
        device=device,
        backbone_lr=args.backbone_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_value=args.gradient_clip,
        save_dir=os.path.join(args.output_dir, 'models', args.model_type)
    )
    
    # Setup mixup if requested
    if args.use_mixup:
        from advanced_augmentation import MixupCutmix
        trainer.mixup_fn = MixupCutmix(alpha=args.mixup_alpha)
        print("Mixup/CutMix enabled")
    
    # Start MLflow run
    run_name = args.run_name or f"{args.model_type}_adv"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            'model_type': args.model_type,
            'use_attention': args.use_attention,
            'loss_type': args.loss_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'backbone_lr': args.backbone_lr,
            'classifier_lr': args.classifier_lr,
            'scheduler': args.scheduler,
            'use_amp': args.use_amp,
            'gradient_accumulation': args.gradient_accumulation,
            'use_advanced_aug': args.use_advanced_aug,
            'use_mixup': args.use_mixup,
            'image_size': args.image_size
        })
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs
        )
        
        # Log final metrics
        mlflow.log_metrics({
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc']),
            'best_val_f1': max(history['val_f1'])
        })
        
        # Save plots
        plot_dir = os.path.join(args.output_dir, 'plots', args.model_type)
        os.makedirs(plot_dir, exist_ok=True)
        
        trainer.plot_learning_curves(history, plot_dir)
        
        print("\nTraining complete!")
        print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
        print(f"Best validation F1: {max(history['val_f1']):.4f}")
        
        return trainer.best_model_path, history


def evaluate_model(args, model, test_loader, device):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    eval_dir = os.path.join(args.output_dir, 'evaluation', args.model_type)
    
    # Run comprehensive evaluation
    results = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=eval_dir
    )
    
    # Test-time augmentation if requested
    if args.use_tta:
        from advanced_augmentation import TestTimeAugmentation
        print("\nRunning test-time augmentation...")
        
        tta = TestTimeAugmentation(model, device, n_augmentations=5)
        all_probs = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                probs = tta.predict(images)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate TTA metrics
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)
        
        tta_acc = accuracy_score(all_labels, preds)
        tta_f1 = f1_score(all_labels, preds)
        
        print(f"\nTest-Time Augmentation Results:")
        print(f"  Accuracy: {tta_acc:.4f}")
        print(f"  F1 Score: {tta_f1:.4f}")
        
        results['tta_metrics'] = {
            'accuracy': tta_acc,
            'f1_score': tta_f1
        }
    
    return results


def main():
    """Main training/evaluation pipeline"""
    
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(args)
    
    # Setup model and loss
    model, criterion = setup_model_and_loss(args, device)
    
    # Evaluation only mode
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for evaluation mode")
        
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        results = evaluate_model(args, model, test_loader, device)
        
        return
    
    # Training mode
    best_model_path, history = train_model(
        args, model, criterion, train_loader, val_loader, device
    )
    
    # Load best model for final evaluation
    print(f"\nLoading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    
    results = evaluate_model(args, model, test_loader, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {best_model_path}")
    print(f"Evaluation results saved to: {os.path.join(args.output_dir, 'evaluation', args.model_type)}")


if __name__ == "__main__":
    main()
