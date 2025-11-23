#!/usr/bin/env python3
"""
Quick Custom CNN Training - 5 Epochs
Optimized for fast training with minimal resources
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import time
from pathlib import Path
import sys
import gc

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class LightweightDeepfakeCNN(nn.Module):
    """Improved CNN with regularization to prevent overfitting"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Simpler architecture to reduce overfitting
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Simpler classifier with more dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeepfakeDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, max_samples=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.max_samples = max_samples
        
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.dataset))
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'] if 'image' in item else item['Image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Handle different label formats
        label_key = 'label' if 'label' in item else 'Label'
        label_value = item[label_key]
        
        # Convert to binary: 0 = Real, 1 = Fake
        if isinstance(label_value, str):
            label = 1 if label_value.lower() in ['fake', 'deepfake'] else 0
        else:
            label = int(label_value)
        
        return image, label

def train_custom_cnn():
    print("=" * 60)
    print("CUSTOM CNN TRAINING - FIXED OVERFITTING")
    print("=" * 60)
    
    # Configuration - Optimized for speed and accuracy
    EPOCHS = 10
    BATCH_SIZE = 64  # Larger batches for faster training
    LEARNING_RATE = 0.001  # Higher initial LR with scheduler
    MAX_SAMPLES = 20000  # 10k real + 10k fake (balanced)
    IMAGE_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n⚙️  Configuration:")
    print(f"   Epochs: {EPOCHS} (with early stopping & LR scheduling)")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE} (adaptive)")
    print(f"   Max Samples: {MAX_SAMPLES} (balanced)")
    print(f"   Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Device: {DEVICE}")
    print(f"   Target Accuracy: 80-85%")
    print(f"   Estimated Time: ~40-50 minutes")
    
    # Data transforms with STRONG augmentation to prevent overfitting
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # More rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Stronger jitter
        transforms.RandomGrayscale(p=0.1),  # Random grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # Random erasing
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset - Using a more challenging dataset
    print(f"\n📥 Loading dataset...")
    try:
        # Try multiple datasets - this one is more realistic
        try:
            print("   Trying FaceForensics++ style dataset...")
            dataset = load_dataset("Leonardo12356/deepfake-detection-dataset", split="train")
            print(f"✅ Dataset loaded: {len(dataset)} total images")
        except:
            print("   Fallback to original dataset...")
            dataset = load_dataset("JamieWithofs/Deepfake-and-real-images", split="train")
            print(f"✅ Dataset loaded: {len(dataset)} total images")
            print("   ⚠️  WARNING: This dataset may be too easy!")
        
        # Shuffle the dataset first
        import random
        indices = list(range(len(dataset)))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        
        # Take subset with balanced real/fake
        print(f"\n   Analyzing dataset balance...")
        real_indices = []
        fake_indices = []
        
        for idx in indices:
            item = dataset[idx]
            label = item.get('label', item.get('Label', 'unknown'))
            if label in ['real', 'Real', 0]:
                real_indices.append(idx)
            elif label in ['fake', 'Fake', 1]:
                fake_indices.append(idx)
            
            # Stop when we have enough of each
            if len(real_indices) >= MAX_SAMPLES//2 and len(fake_indices) >= MAX_SAMPLES//2:
                break
        
        # Balance the dataset
        target_per_class = min(len(real_indices), len(fake_indices), MAX_SAMPLES//2)
        balanced_indices = real_indices[:target_per_class] + fake_indices[:target_per_class]
        random.shuffle(balanced_indices)
        
        dataset = dataset.select(balanced_indices)
        print(f"📊 Balanced dataset: {len(dataset)} samples ({target_per_class} real, {target_per_class} fake)")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Split dataset (80-20 split)
    split_idx = int(len(dataset) * 0.8)
    
    print(f"\n📊 Creating datasets...")
    train_dataset = DeepfakeDataset(
        dataset.select(range(split_idx)),
        transform=transform_train
    )
    val_dataset = DeepfakeDataset(
        dataset.select(range(split_idx, MAX_SAMPLES)),
        transform=transform_val
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Model
    print(f"\n🧠 Creating model...")
    model = LightweightDeepfakeCNN(num_classes=2).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training
    print(f"\n🚀 Starting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    start_epoch = 0
    output_dir = Path("runs/models/custom_lightweight")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / "checkpoint.pth"
    history_path = output_dir / "training_progress.txt"
    
    # Early stopping parameters
    patience = 3
    patience_counter = 0
    best_epoch = 0
    
    # Resume from checkpoint if it exists
    if checkpoint_path.exists():
        print(f"\n📂 Found checkpoint! Loading from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            patience_counter = checkpoint.get('patience_counter', 0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"✅ Resuming from epoch {start_epoch + 1}/{EPOCHS}, best val acc: {best_val_acc:.2f}%")
            print(f"   Patience: {patience_counter}/{patience}, Best epoch: {best_epoch + 1}")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("   Starting fresh training...")
            start_epoch = 0
    else:
        # Write initial progress file
        with open(history_path, 'w') as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {EPOCHS}\n")
            f.write(f"Samples: {MAX_SAMPLES}\n")
            f.write("-" * 60 + "\n")
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n📈 Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                batch_acc = 100. * train_correct / train_total
                print(f"   Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.4f} | "
                      f"Acc: {batch_acc:.2f}%")
                
                # Clear memory periodically
                if (batch_idx + 1) % 100 == 0:
                    gc.collect()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n   📊 Epoch {epoch + 1} Results:")
        print(f"      Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"      Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"      Time: {epoch_time:.2f}s")
        
        # Check for overfitting warning
        if train_acc > 98 and val_acc < 85:
            print(f"      ⚠️  WARNING: Possible overfitting (train >> val)")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model_cpu.pth")
            print(f"      ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"      📊 No improvement ({patience_counter}/{patience})")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"\n      🛑 Early stopping triggered! No improvement for {patience} epochs")
                print(f"      Best model from epoch {best_epoch + 1} with {best_val_acc:.2f}% accuracy")
                break
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'patience_counter': patience_counter,
            'best_epoch': best_epoch,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"      💾 Checkpoint saved (epoch {epoch + 1}/{EPOCHS})")
        
        # Update progress file
        with open(history_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%\n")
        
        # Clear memory after each epoch
        gc.collect()
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Total Training Time: {total_time/60:.2f} minutes")
    print(f"   Model saved to: {output_dir / 'best_model_cpu.pth'}")
    print(f"   Checkpoint saved to: {checkpoint_path}")
    print("=" * 60)
    
    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"\n🗑️  Checkpoint deleted (training complete)")
    
    # Update progress file
    with open(history_path, 'a') as f:
        f.write("-" * 60 + "\n")
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Total time: {total_time/60:.2f} minutes\n")
    
    # Save model info
    info = {
        'epochs': EPOCHS,
        'best_val_acc': best_val_acc,
        'total_time': total_time,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'total_params': total_params
    }
    
    import json
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ Training info saved to: {output_dir / 'training_info.json'}")
    print("\n🚀 You can now use this model in the app!")

if __name__ == "__main__":
    train_custom_cnn()
