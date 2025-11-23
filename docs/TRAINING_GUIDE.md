# Advanced Training Guide - Deepfake Detection

This guide covers the advanced training pipeline with state-of-the-art techniques for improved deepfake detection accuracy.

## 🚀 Quick Start

### Basic Training (EfficientNet with Advanced Features)

```bash
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type efficientnet \
    --use_attention \
    --use_advanced_aug \
    --use_mixup \
    --use_amp \
    --num_epochs 30 \
    --batch_size 16
```

### Advanced Training (Swin Transformer with All Features)

```bash
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type swin \
    --use_attention \
    --use_advanced_aug \
    --use_mixup \
    --use_tta \
    --use_amp \
    --loss_type focal \
    --num_epochs 50 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --early_stopping_patience 15
```

## 📊 What's New

### 1. Advanced Data Augmentation (`advanced_augmentation.py`)

- **RandAugment**: Automated augmentation policy with 12 operations
- **Cutout**: Random masking for robustness
- **Mixup/CutMix**: Mixing training samples for better generalization
- **Deepfake-Specific Augmentations**: JPEG compression artifacts, blur, noise
- **Test-Time Augmentation (TTA)**: Multiple predictions averaged during inference

### 2. Enhanced Model Architectures (`enhanced_models.py`)

- **Attention Mechanisms**:
  - Squeeze-and-Excitation (SE) blocks for channel attention
  - Convolutional Block Attention Module (CBAM) for spatial+channel attention
- **Improved Classifiers**:
  - Multi-layer heads with batch normalization
  - Residual connections
  - Optional uncertainty estimation
- **Advanced Loss Functions**:
  - Focal Loss for handling class imbalance
  - Label Smoothing for better calibration

### 3. Advanced Training Techniques (`train_advanced.py`)

- **Mixed Precision Training (AMP)**: 2x faster training, reduced memory
- **Gradient Accumulation**: Effective larger batch sizes
- **Differential Learning Rates**: Lower LR for pretrained backbone
- **Advanced Schedulers**:
  - Cosine annealing with warmup
  - ReduceLROnPlateau with patience
  - OneCycleLR for superconvergence
- **Early Stopping**: Prevents overfitting
- **Comprehensive Metrics**: Accuracy, F1, AUC-ROC, AUC-PR
- **Visualization**: Learning curves, confusion matrix, ROC/PR curves

### 4. Ensemble Methods (`ensemble_inference.py`)

- **Weighted Ensemble**: Combine multiple models with learned weights
- **Voting Mechanisms**: Hard/soft voting for robustness
- **Uncertainty Estimation**: Monte Carlo dropout for confidence
- **Calibration**: Temperature scaling for better probabilities
- **Adaptive Thresholding**: Optimize decision threshold for target metric

### 5. Comprehensive Evaluation (`evaluation_tools.py`)

- **Detailed Metrics**: All classification metrics + calibration
- **Error Analysis**: Identify patterns in false positives/negatives
- **Confidence Analysis**: High-confidence errors detection
- **Visualizations**: 6-panel evaluation dashboard

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── fake/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## 🎯 Command Line Arguments

### Data Arguments

- `--data_dir`: Path to dataset directory (required)
- `--batch_size`: Batch size (default: 16)
- `--num_workers`: Data loading workers (default: 4)

### Model Arguments

- `--model_type`: Architecture choice (`efficientnet` or `swin`)
- `--use_attention`: Enable attention mechanisms
- `--dropout`: Dropout rate (default: 0.3)
- `--image_size`: Input size (default: 224)

### Training Arguments

- `--num_epochs`: Training epochs (default: 30)
- `--backbone_lr`: Backbone learning rate (default: 1e-5)
- `--classifier_lr`: Classifier learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--scheduler`: LR scheduler (`cosine`, `plateau`, `onecycle`)
- `--warmup_epochs`: Warmup epochs (default: 3)

### Loss Arguments

- `--loss_type`: Loss function (`bce`, `focal`, `label_smoothing`)
- `--focal_alpha`: Focal loss alpha (default: 0.25)
- `--focal_gamma`: Focal loss gamma (default: 2.0)
- `--label_smoothing`: Smoothing factor (default: 0.1)

### Augmentation Arguments

- `--use_advanced_aug`: Enable RandAugment + deepfake augmentations
- `--use_mixup`: Enable Mixup/CutMix
- `--mixup_alpha`: Mixup alpha (default: 0.2)
- `--use_tta`: Test-time augmentation

### Advanced Training Arguments

- `--use_amp`: Mixed precision training
- `--gradient_accumulation`: Accumulation steps (default: 1)
- `--early_stopping_patience`: Patience epochs (default: 10)
- `--gradient_clip`: Gradient clipping (default: 1.0)

### Output Arguments

- `--output_dir`: Output directory (default: ./runs)
- `--experiment_name`: MLflow experiment name
- `--run_name`: MLflow run name

### Evaluation Arguments

- `--eval_only`: Evaluation mode only
- `--checkpoint`: Model checkpoint path

## 💡 Training Tips

### 1. Start with Recommended Settings

**For EfficientNet-B3:**
```bash
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type efficientnet \
    --use_attention \
    --use_advanced_aug \
    --use_mixup \
    --use_amp \
    --loss_type focal \
    --num_epochs 30 \
    --batch_size 16 \
    --scheduler cosine \
    --early_stopping_patience 10
```

**For Swin Transformer:**
```bash
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type swin \
    --use_attention \
    --use_advanced_aug \
    --use_mixup \
    --use_amp \
    --loss_type focal \
    --num_epochs 50 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --scheduler cosine \
    --early_stopping_patience 15
```

### 2. Memory Management

If you encounter OOM (Out of Memory) errors:

- Reduce `--batch_size` (e.g., 8 → 4)
- Increase `--gradient_accumulation` to maintain effective batch size
- Reduce `--image_size` (e.g., 224 → 192)
- Disable TTA during training (`--use_tta` only for final evaluation)

### 3. Hyperparameter Tuning

**Learning Rates:**
- Backbone: Start with 1e-5, reduce if unstable
- Classifier: Start with 1e-4, can increase to 1e-3

**Scheduler Choice:**
- `cosine`: Most stable, recommended for long training
- `plateau`: Good for adaptive learning
- `onecycle`: Fast convergence but requires tuning

**Augmentation Strength:**
- Start with `--use_advanced_aug` only
- Add `--use_mixup` if accuracy plateaus
- Adjust `--mixup_alpha` (0.1-0.4 range)

### 4. Expected Performance

| Model | Configuration | Expected Accuracy |
|-------|--------------|------------------|
| EfficientNet-B3 | Basic | 96-97% |
| EfficientNet-B3 | + Attention + Focal Loss | 97-98% |
| EfficientNet-B3 | + All Features | 98-99% |
| Swin-Base | Basic | 97-98% |
| Swin-Base | + All Features | 99%+ |

## 📈 Monitoring Training

### MLflow Dashboard

1. Start MLflow server:
```bash
mlflow ui --port 5000
```

2. Open browser: `http://localhost:5000`

3. View:
   - Real-time metrics (loss, accuracy, F1)
   - Learning rate changes
   - Model checkpoints
   - Training parameters

### Training Logs

Watch training progress:
```bash
tail -f runs/models/efficientnet/training.log
```

### Visualizations

After training, check:
- `runs/plots/<model_type>/learning_curves.png`: Training progress
- `runs/plots/<model_type>/confusion_matrix.png`: Final confusion matrix
- `runs/plots/<model_type>/roc_curve.png`: ROC curve
- `runs/plots/<model_type>/pr_curve.png`: Precision-Recall curve

## 🔍 Evaluation Only Mode

Evaluate a trained model:

```bash
python training/train_unified.py \
    --eval_only \
    --checkpoint runs/models/efficientnet/best_model.pth \
    --data_dir /path/to/dataset \
    --model_type efficientnet \
    --use_attention \
    --use_tta
```

Results saved to: `runs/evaluation/<model_type>/`

## 🎭 Ensemble Inference

Combine multiple models for best accuracy:

```python
from ensemble_inference import EnsemblePredictor

# Load models
models = [model1, model2, model3]
weights = [0.4, 0.35, 0.25]  # Based on validation performance

# Create ensemble
ensemble = EnsemblePredictor(models, device, weights=weights)

# Predict
prediction, confidence = ensemble.predict(image_tensor)
```

## 🐛 Troubleshooting

### Issue: Training is slow

**Solutions:**
- Enable `--use_amp` for 2x speedup
- Increase `--num_workers` for data loading
- Use smaller `--image_size`
- Check GPU utilization: `nvidia-smi`

### Issue: Model not learning

**Solutions:**
- Check data labels are correct
- Increase learning rates
- Reduce regularization (dropout, weight_decay)
- Try different loss function
- Verify data augmentation isn't too aggressive

### Issue: Overfitting (train acc >> val acc)

**Solutions:**
- Enable `--use_advanced_aug` and `--use_mixup`
- Increase `--dropout`
- Add label smoothing: `--loss_type label_smoothing`
- Reduce model complexity
- Early stopping will handle this automatically

### Issue: Underfitting (low train acc)

**Solutions:**
- Increase model capacity: use `swin` instead of `efficientnet`
- Increase learning rates
- Train longer: more epochs
- Reduce regularization
- Check if data quality is sufficient

## 📚 Additional Resources

- **Model Architectures**: See `enhanced_models.py` for implementation details
- **Augmentation Policies**: See `advanced_augmentation.py` for all transforms
- **Training Loop**: See `train_advanced.py` for training logic
- **Evaluation Metrics**: See `evaluation_tools.py` for all metrics

## 🎓 Best Practices

1. **Always use validation set** for hyperparameter tuning
2. **Test set is sacred** - only evaluate final model
3. **Start simple** - baseline → add features incrementally
4. **Monitor overfitting** - use early stopping
5. **Ensemble multiple runs** - train 3-5 models, ensemble best ones
6. **Use TTA for final evaluation** - improves accuracy by 0.5-1%
7. **Calibrate predictions** - use temperature scaling for better probabilities

## 📞 Support

For issues or questions:
1. Check this guide first
2. Review example commands
3. Examine training logs
4. Check GPU memory usage
5. Verify dataset structure

Happy training! 🚀
