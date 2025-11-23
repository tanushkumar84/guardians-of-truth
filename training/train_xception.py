"""
Train XceptionNet from scratch for Deepfake Detection
Third ensemble component alongside EfficientNet and Swin Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import time
import json
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from datetime import datetime

# Import XceptionNet
from xception_model import create_xception, count_parameters


class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir: Root directory with 'real' and 'fake' subdirectories
            transform: Transform to apply to images
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        # Load dataset using ImageFolder
        self.dataset = datasets.ImageFolder(root=str(self.root_dir), transform=transform)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class XceptionTrainer:
    """Trainer for XceptionNet from scratch"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir='runs/models/xception',
        experiment_name='xception_from_scratch'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f'xception_epoch{epoch}_val{val_acc:.4f}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved: {best_path}")
            
            # Save CPU version for deployment
            cpu_model = create_xception(num_classes=2, dropout_rate=0.5)
            cpu_model.load_state_dict(self.model.state_dict())
            cpu_model = cpu_model.cpu()
            cpu_path = self.save_dir / 'best_model_cpu.pth'
            torch.save(cpu_model.state_dict(), cpu_path)
            print(f"💻 CPU model saved: {cpu_path}")
    
    def train(self, num_epochs, patience=5):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
        """
        print("\n" + "="*80)
        print(f"🚀 Starting XceptionNet Training FROM SCRATCH")
        print("="*80)
        print(f"Model: XceptionNet")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*80 + "\n")
        
        # Start MLflow run
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=f"xception_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model", "XceptionNet")
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", self.train_loader.batch_size)
            mlflow.log_param("optimizer", type(self.optimizer).__name__)
            mlflow.log_param("learning_rate", self.optimizer.param_groups[0]['lr'])
            mlflow.log_param("parameters", count_parameters(self.model))
            
            epochs_without_improvement = 0
            
            for epoch in range(1, num_epochs + 1):
                start_time = time.time()
                
                # Train
                train_loss, train_acc = self.train_epoch(epoch)
                
                # Validate
                val_loss, val_acc = self.validate(epoch)
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rates'].append(current_lr)
                
                # Log to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                
                # Print epoch summary
                epoch_time = time.time() - start_time
                print(f"\n{'='*80}")
                print(f"Epoch {epoch}/{num_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
                print(f"  Learning Rate: {current_lr:.6f}")
                print(f"  Time: {epoch_time:.2f}s")
                
                # Check if best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    print(f"  🎉 New best validation accuracy: {val_acc:.2f}% (previous: {self.best_val_acc:.2f}%)")
                    self.best_val_acc = val_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f"  ⏳ No improvement for {epochs_without_improvement} epoch(s)")
                
                print(f"{'='*80}\n")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_acc, is_best=is_best)
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                    print(f"No improvement for {patience} consecutive epochs")
                    break
            
            # Save training history
            history_path = self.save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Log final metrics
            mlflow.log_metric("best_val_acc", self.best_val_acc)
            mlflow.log_artifact(str(history_path))
            
            print("\n" + "="*80)
            print("✅ Training completed!")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            print(f"Models saved in: {self.save_dir}")
            print("="*80 + "\n")


def get_data_transforms():
    """Get data transforms for training and validation"""
    
    # XceptionNet expects 299x299 input
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def main():
    """Main training function"""
    
    # ========== Configuration ==========
    CONFIG = {
        'data_dir': 'path/to/dataset',  # UPDATE THIS: Your dataset directory
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'dropout_rate': 0.5,
        'patience': 5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n" + "="*80)
    print("🎯 XceptionNet Training Configuration")
    print("="*80)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    # ========== Device ==========
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # ========== Data Transforms ==========
    train_transform, val_transform = get_data_transforms()
    
    # ========== Dataset ==========
    # IMPORTANT: Update data_dir to your dataset location
    # Expected structure:
    # data_dir/
    #   train/
    #     real/
    #     fake/
    #   val/
    #     real/
    #     fake/
    
    try:
        train_dataset = datasets.ImageFolder(
            root=os.path.join(CONFIG['data_dir'], 'train'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(CONFIG['data_dir'], 'val'),
            transform=val_transform
        )
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        print(f"Classes: {train_dataset.classes}\n")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\n⚠️  Please update CONFIG['data_dir'] with your dataset path!")
        print("Expected structure:")
        print("  data_dir/")
        print("    train/")
        print("      real/")
        print("      fake/")
        print("    val/")
        print("      real/")
        print("      fake/\n")
        return
    
    # ========== DataLoaders ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # ========== Model ==========
    print("🔧 Creating XceptionNet model...")
    model = create_xception(
        num_classes=2,
        dropout_rate=CONFIG['dropout_rate']
    )
    print(f"✅ Model created with {count_parameters(model):,} parameters\n")
    
    # ========== Loss Function ==========
    criterion = nn.CrossEntropyLoss()
    
    # ========== Optimizer ==========
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # ========== Learning Rate Scheduler ==========
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )
    
    # ========== Trainer ==========
    trainer = XceptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir='runs/models/xception',
        experiment_name='xception_from_scratch'
    )
    
    # ========== Train ==========
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        patience=CONFIG['patience']
    )
    
    print("\n✅ Training script completed successfully!")
    print("📁 Model files saved in: runs/models/xception/")
    print("   - best_model.pth (full checkpoint)")
    print("   - best_model_cpu.pth (CPU deployment)")
    print("   - training_history.json (metrics)")


if __name__ == "__main__":
    main()
