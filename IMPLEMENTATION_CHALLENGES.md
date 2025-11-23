# Implementation Details & Challenges

## Project Overview
**Deepfake Detection System** - Multi-model ensemble combining EfficientNet-B3, Swin Transformer, and Custom CNN for real/fake image and video analysis.

---

## Key Implementation Details

### 1. **Model Architecture & Training**

#### EfficientNet-B3 (Primary Model)
- **Configuration**: Pre-trained on ImageNet, fine-tuned on deepfake dataset
- **Input**: 300×300×3 RGB images
- **Custom Head**: Global Average Pooling → FC(1536) → ReLU → Dropout(0.5) → FC(512) → FC(2)
- **Training Strategy**:
  - Loss: CrossEntropyLoss with label smoothing (0.1)
  - Optimizer: AdamW with differential learning rates (backbone: 1e-5, classifier: 1e-4)
  - Scheduler: ReduceLROnPlateau (patience=3)
  - Early stopping: patience=5
- **Challenge**: Memory management with 12M parameters required gradient checkpointing

#### Swin Transformer Base
- **Configuration**: Hierarchical vision transformer with shifted windows
- **Input**: 224×224×3 RGB images
- **Advanced Augmentation**: RandAugment + Mixup (α=0.2) + CutMix (α=1.0)
- **Training Strategy**:
  - Loss: Focal Loss (γ=2.0, α=0.25) for class imbalance
  - Optimizer: AdamW (lr=5e-5, weight_decay=0.05)
  - Scheduler: Cosine annealing with warmup
- **Challenge**: 88M parameters caused OOM errors → solved with mixed precision training (FP16)

#### Custom Lightweight CNN
- **Architecture**: 4 conv blocks (32→64→128→256 channels) + GAP + 2 FC layers
- **Parameters**: Only 430K → 1.7MB model size
- **Purpose**: Fast inference for resource-constrained environments
- **Challenge**: Lower accuracy (81-83%) required careful tuning of ensemble weights

### 2. **Face Detection Pipeline**

#### MediaPipe Integration
- **Model**: `short_range` face detection model
- **Threshold**: 0.5 confidence to balance precision/recall
- **Padding Strategy**: 20% bbox expansion to capture context
- **Issue Faced**: 
  - **Problem**: False negatives on profile faces and occluded faces
  - **Solution**: Implemented fallback to full image analysis when no face detected
  - **Problem**: BGR vs RGB color space mismatch
  - **Solution**: Added explicit `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` conversion

### 3. **Ensemble Strategy**

#### Weighted Voting System
```
Final_Score = 0.4 × EfficientNet + 0.4 × Swin + 0.2 × Custom_CNN
Verdict = "FAKE" if Final_Score > 0.5 else "REAL"
```

**Weight Selection Process**:
- Tested combinations: [0.33, 0.33, 0.34], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]
- Selected [0.4, 0.4, 0.2] based on validation accuracy (improved from 91.2% to 93.1%)
- **Challenge**: Single model failures could skew results → added confidence thresholding

### 4. **Video Processing**

#### Frame Extraction Strategy
- **Uniform Sampling**: Extract N frames evenly distributed across video duration
- **Default**: 10 frames for videos <30s, 20 frames for longer videos
- **Temporal Aggregation**: Average probabilities across all frames
- **Issue Faced**:
  - **Problem**: Videos with scene changes caused inconsistent predictions
  - **Solution**: Implemented per-frame confidence weighting (higher weight for high-confidence frames)
  - **Problem**: Memory overflow with 4K videos
  - **Solution**: Added frame resizing before processing + batch processing

### 5. **Streamlit UI Challenges**

#### Dark Theme Implementation
- **Goal**: Cyber-aesthetic with animated starfield background
- **CSS Challenges**:
  1. **Text Invisibility**: Dark text on dark background
     - Tried: Basic `color: white` → Failed (Streamlit overrides)
     - Tried: Inline styles → Failed (low specificity)
     - **Solution**: Aggressive CSS with `!important` + `[data-testid]` targeting + z-index layering
  
  2. **Radio Button Visibility**: Buttons completely hidden
     - **Solution**: Added `background: rgba(23,198,216,0.25)` + `border: 2px solid #17c6d8` + larger fonts
  
  3. **File Uploader Invisible**: Browse button hidden on black background
     - **Solution**: Cyan dashed border (3px) + semi-transparent white background + black text on cyan button

#### Final CSS Stack
```css
z-index: content(1000) >> UI elements(10) >> background(0)
All text: color: #ffffff !important
Interactive elements: cyan accents (#17c6d8)
Hover effects: brightness(1.2) for feedback
```

### 6. **Model Deployment**

#### Kaggle Model Hosting
- **Why Kaggle**: Free hosting for large model files (50MB + 338MB + 1.7MB)
- **Download Strategy**: `setup.sh` script using Kaggle API on first run
- **Issue Faced**:
  - **Problem**: Streamlit Cloud doesn't persist uploaded files between deployments
  - **Solution**: Models downloaded to `runs/models/` on container startup
  - **Problem**: API token security in public repos
  - **Solution**: `.streamlit/secrets.toml` + environment variables + `.gitignore` protection

#### Resource Optimization
- **CPU-Only Deployment**: All models converted to CPU-compatible `.pth` files
- **Model Caching**: `@st.cache_resource` to prevent reloading on every interaction
- **Session State**: Persistent user data across re-runs
- **Challenge**: 1GB RAM limit on Streamlit Cloud free tier
  - **Solution**: Sequential model loading + garbage collection after inference

### 7. **Preprocessing Pipeline**

#### Model-Specific Transforms
- **EfficientNet**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Swin**: Same as EfficientNet (transfer learning from ImageNet)
- **Custom CNN**: Simple [0, 1] normalization
- **Issue Faced**:
  - **Problem**: Inconsistent input sizes caused dimension errors
  - **Solution**: Separate preprocessing functions per model with explicit resize + center crop

---

## Major Issues Encountered & Solutions

### Issue #1: Out of Memory Errors
**Problem**: Training Swin Transformer with 88M parameters on limited GPU memory  
**Symptoms**: CUDA OOM errors during forward pass  
**Solutions Applied**:
1. Mixed precision training (FP16) → 40% memory reduction
2. Gradient accumulation (effective batch size 32 from 8×4)
3. Smaller batch size during validation
4. Gradient checkpointing for Swin blocks

### Issue #2: Class Imbalance
**Problem**: Dataset had 60% fake, 40% real → model biased toward "FAKE" predictions  
**Solutions**:
1. Focal Loss for Swin (focuses on hard examples)
2. Label smoothing for EfficientNet
3. Class-weighted sampling during training
4. Ensemble thresholding tuned to 0.5 (neutral)

### Issue #3: UI Text Completely Invisible
**Problem**: Dark theme + Streamlit's default styles = nothing visible  
**Root Cause**: Streamlit applies inline styles with high specificity  
**Solution Evolution**:
- Attempt 1: Standard CSS → Failed (overridden)
- Attempt 2: Increased z-index → Failed (text still black)
- Attempt 3: `!important` flags → Partial success
- **Final**: Nuclear CSS with `[data-baseweb]` and `[data-testid]` targeting + z-index 1000 + explicit white colors

### Issue #4: Video Processing Timeouts
**Problem**: Large videos (>100MB) caused 30s timeout on Streamlit Cloud  
**Solutions**:
1. Frame sampling instead of processing all frames
2. Progress bar for user feedback
3. Maximum 20 frames limit
4. Early stopping if 15 consecutive frames give same verdict

### Issue #5: Face Detection Failures
**Problem**: MediaPipe missed 15-20% of faces (profile views, masks, occlusions)  
**Solutions**:
1. Lowered confidence threshold from 0.7 to 0.5
2. Fallback to full-frame analysis when no face detected
3. User notification when face not found
4. Added 20% padding to prevent face cropping

### Issue #6: Model Version Compatibility
**Problem**: Models trained with PyTorch 1.x wouldn't load in PyTorch 2.x environment  
**Solution**: Re-saved all models using `torch.save()` with `_use_new_zipfile_serialization=True`

---

## Performance Metrics

| Model | Accuracy | Size | Inference Time | Memory |
|-------|----------|------|----------------|--------|
| EfficientNet-B3 | 90-92% | 50MB | ~0.3s | 400MB |
| Swin Transformer | 92-94% | 338MB | ~0.5s | 1.2GB |
| Custom CNN | 81-83% | 1.7MB | ~0.1s | 150MB |
| **Ensemble** | **93-94%** | **390MB** | **~0.9s** | **1.75GB** |

**Production Optimization**:
- CPU inference: ~2.5s per image (acceptable for user-facing app)
- Video (10 frames): ~25s average
- Streamlit Cloud deployment: 1GB RAM, works with sequential loading

---

## Lessons Learned

1. **Model Selection**: Ensemble of diverse architectures (CNN + ViT) outperforms single model
2. **UI/UX Matters**: Spent 30% of time on CSS fixes for dark theme visibility
3. **Resource Constraints**: Free hosting requires aggressive optimization (caching, model compression)
4. **Face Detection**: Not 100% reliable → always provide fallback options
5. **Deployment Early**: Found Streamlit Cloud RAM limits only after attempting deployment
6. **Documentation**: Comprehensive guides (DEPLOYMENT.md, DIAGRAMS.txt) saved time explaining to users

---

**Project Status**: ✅ Complete and deployed  
**Total Development Time**: ~3 weeks  
**Lines of Code**: ~3,500 (excluding generated files)
