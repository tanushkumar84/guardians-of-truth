# Quick Reference - Advanced Training Commands

## 🚀 Training Commands

### Recommended Settings

#### EfficientNet-B3 (Good balance of speed and accuracy)
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
    --backbone_lr 1e-5 \
    --classifier_lr 1e-4 \
    --scheduler cosine \
    --warmup_epochs 3 \
    --early_stopping_patience 10 \
    --output_dir ./runs \
    --experiment_name deepfake_efficientnet
```

#### Swin Transformer (Best accuracy)
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
    --backbone_lr 1e-5 \
    --classifier_lr 1e-4 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --early_stopping_patience 15 \
    --output_dir ./runs \
    --experiment_name deepfake_swin
```

### Fast Prototyping (Lower memory, faster iterations)
```bash
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type efficientnet \
    --use_amp \
    --num_epochs 15 \
    --batch_size 32 \
    --image_size 192 \
    --scheduler onecycle
```

### Maximum Accuracy (All features enabled)
```bash
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type swin \
    --use_attention \
    --use_advanced_aug \
    --use_mixup \
    --mixup_alpha 0.3 \
    --use_tta \
    --use_amp \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --num_epochs 100 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --backbone_lr 5e-6 \
    --classifier_lr 5e-5 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --early_stopping_patience 20 \
    --gradient_clip 1.0
```

## 🔍 Evaluation Commands

### Evaluate Single Model
```bash
python training/train_unified.py \
    --eval_only \
    --checkpoint runs/models/efficientnet/best_model.pth \
    --data_dir /path/to/dataset \
    --model_type efficientnet \
    --use_attention
```

### Evaluate with Test-Time Augmentation
```bash
python training/train_unified.py \
    --eval_only \
    --checkpoint runs/models/swin/best_model.pth \
    --data_dir /path/to/dataset \
    --model_type swin \
    --use_attention \
    --use_tta \
    --batch_size 8
```

## 📊 MLflow Monitoring

### Start MLflow UI
```bash
mlflow ui --port 5000
```
Then open: http://localhost:5000

### List all experiments
```bash
mlflow experiments list
```

### View specific run
```bash
mlflow runs describe --run-id <run_id>
```

## 🎭 Ensemble Prediction (Python)

```python
from ensemble_inference import EnsemblePredictor
import torch

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = load_model('runs/models/efficientnet/best_model.pth', device)
model2 = load_model('runs/models/swin/best_model.pth', device)

# Create ensemble with custom weights
ensemble = EnsemblePredictor(
    models=[model1, model2],
    device=device,
    weights=[0.45, 0.55]  # Slightly favor Swin
)

# Predict
image = load_and_preprocess_image('test.jpg')
prediction, confidence = ensemble.predict(image, method='weighted')

print(f"Prediction: {'FAKE' if prediction > 0.5 else 'REAL'}")
print(f"Confidence: {confidence:.4f}")
```

## 🐛 Common Issues and Solutions

### Out of Memory
```bash
# Reduce batch size and use gradient accumulation
python training/train_unified.py \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --image_size 192
```

### Training Too Slow
```bash
# Enable mixed precision and increase workers
python training/train_unified.py \
    --use_amp \
    --num_workers 8 \
    --batch_size 32
```

### Overfitting
```bash
# Enable all regularization techniques
python training/train_unified.py \
    --use_advanced_aug \
    --use_mixup \
    --mixup_alpha 0.3 \
    --dropout 0.5 \
    --weight_decay 1e-3 \
    --loss_type label_smoothing \
    --label_smoothing 0.1
```

### Underfitting
```bash
# Increase model capacity and reduce regularization
python training/train_unified.py \
    --model_type swin \
    --use_attention \
    --dropout 0.2 \
    --weight_decay 1e-5 \
    --num_epochs 100
```

## 📁 File Locations

### Training Outputs
- Models: `runs/models/<model_type>/best_model.pth`
- Plots: `runs/plots/<model_type>/`
- MLflow logs: `runs/mlruns/`

### Evaluation Outputs
- Report: `runs/evaluation/<model_type>/evaluation_report.txt`
- Metrics: `runs/evaluation/<model_type>/evaluation_report.json`
- Plots: `runs/evaluation/<model_type>/comprehensive_evaluation.png`

## ⚙️ Key Hyperparameters

| Parameter | EfficientNet | Swin | Notes |
|-----------|-------------|------|-------|
| `--batch_size` | 16-32 | 8-16 | Swin needs more memory |
| `--image_size` | 224-300 | 224 | EfficientNet can use 300 |
| `--backbone_lr` | 1e-5 | 1e-5 | Low for pretrained |
| `--classifier_lr` | 1e-4 | 1e-4 | Higher for new layers |
| `--num_epochs` | 30-50 | 50-100 | Swin benefits from longer |
| `--warmup_epochs` | 3-5 | 5-10 | More for complex models |
| `--dropout` | 0.3-0.5 | 0.3-0.4 | Adjust based on overfitting |
| `--mixup_alpha` | 0.2-0.4 | 0.2-0.3 | Lower = less mixing |
| `--focal_alpha` | 0.25 | 0.25 | For class imbalance |
| `--focal_gamma` | 2.0 | 2.0 | Higher = more focus on hard examples |

## 🎯 Expected Training Times

*On NVIDIA T4 GPU:*

| Configuration | Time per Epoch | Total Time (30 epochs) |
|--------------|----------------|----------------------|
| EfficientNet, batch=16, no AMP | ~5 min | ~2.5 hours |
| EfficientNet, batch=16, AMP | ~3 min | ~1.5 hours |
| Swin, batch=8, no AMP | ~8 min | ~4 hours |
| Swin, batch=8, AMP | ~5 min | ~2.5 hours |

*On CPU (not recommended for training):*
- 10-20x slower than GPU

## 💡 Pro Tips

1. **Start with EfficientNet** - faster iterations, good results
2. **Use AMP always** - 2x speedup, minimal accuracy loss
3. **Monitor MLflow** - catch issues early
4. **Save checkpoints frequently** - training can be interrupted
5. **Use TTA for final eval** - 0.5-1% accuracy boost
6. **Ensemble 3-5 models** - best single-model performance
7. **Calibrate predictions** - better confidence estimates
8. **Test on diverse data** - avoid overfitting to specific datasets

## 📞 Getting Help

1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed documentation
2. Review training logs: `tail -f runs/models/<model>/training.log`
3. Check GPU usage: `nvidia-smi` or `watch -n 1 nvidia-smi`
4. Verify data loading: Print first batch to check labels/transforms
5. Compare to baseline: Run without advanced features first
