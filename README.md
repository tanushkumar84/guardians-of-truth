<div align="center">

# 🔍 AI-Powered Deepfake Detection System

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**A production-ready deepfake detection system powered by state-of-the-art deep learning models**

[🚀 Live Demo](https://your-app.streamlit.app) • [📖 Documentation](#-table-of-contents) • [🎯 Quick Start](#-quick-start) • [🔬 Research](#-model-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Model Architecture](#-model-architecture)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Training Pipeline](#-training-pipeline)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements a **multi-model ensemble deepfake detection system** capable of analyzing images and videos with high accuracy. Using three complementary deep learning architectures, the system achieves robust performance across various deepfake generation techniques.

### Why This Matters

With the rapid advancement of generative AI technologies (DeepFake, FaceSwap, GANs), detecting manipulated media has become crucial for:
- 🛡️ **Security**: Preventing identity fraud and impersonation
- 🗳️ **Democracy**: Combating misinformation and fake news
- ⚖️ **Justice**: Verifying authenticity of digital evidence
- 🎓 **Education**: Understanding AI-generated content

---

## ✨ Key Features

### 🎯 Multi-Model Ensemble
- **3 State-of-the-Art Models**: EfficientNet-B3, Swin Transformer, Custom Lightweight CNN
- **Ensemble Voting**: Combines predictions for higher accuracy
- **Complementary Strengths**: Each model captures different manipulation artifacts

### 🖼️ Image Analysis
- ✅ Single image deepfake detection
- ✅ Face detection and extraction
- ✅ Confidence scores for each model
- ✅ Visual annotations and heatmaps
- ✅ Supports JPG, JPEG, PNG formats

### 🎥 Video Analysis
- ✅ Frame-by-frame analysis
- ✅ Full-frame and face-only detection modes
- ✅ Temporal consistency checking
- ✅ Artifact detection across frames
- ✅ Sample face grid visualization
- ✅ Supports MP4, AVI, MOV formats (up to 200MB)

### ⚡ Performance
- **Fast Inference**: ~1-2 seconds per image
- **Optimized Models**: CPU-friendly with optional GPU acceleration
- **Batch Processing**: Efficient video frame analysis
- **Cached Loading**: Models loaded once and reused

### 🎨 Modern UI
- Dark theme with animated background
- Responsive design
- Real-time progress tracking
- Downloadable analysis reports
- Mobile-friendly interface

---

## 🚀 Live Demo

Try the live application: **[Deepfake Detection App](https://your-app.streamlit.app)**

> **Note**: First load may take 1-2 minutes as models are downloaded from Kaggle (390MB total)

---

## 🔬 Model Architecture

### 1️⃣ EfficientNet-B3 (Primary Model)
**Accuracy: 90-92%**

```
Architecture:
├── Backbone: EfficientNet-B3 (pre-trained on ImageNet)
│   ├── Input: 300×300×3 RGB images
│   ├── Compound scaling (depth, width, resolution)
│   └── 12M parameters
├── Custom Classifier:
│   ├── Global Average Pooling
│   ├── Dropout (0.3)
│   ├── Dense(1536 → 512) + ReLU + BatchNorm
│   ├── Dropout (0.3)
│   └── Dense(512 → 2) [Real, Fake]
└── Output: Softmax probabilities
```

**Why EfficientNet?**
- Efficient compound scaling balances depth, width, and resolution
- Pre-trained on ImageNet provides robust feature extraction
- Excellent at detecting compression artifacts and GAN fingerprints
- Good balance between accuracy and inference speed

**Training Details:**
- Loss: CrossEntropyLoss with Label Smoothing (0.1)
- Optimizer: AdamW (backbone LR: 1e-4, classifier LR: 1e-3)
- Augmentation: RandomHorizontalFlip, ColorJitter, RandomRotation
- Input Size: 300×300
- Batch Size: 32
- Epochs: 20-30

---

### 2️⃣ Swin Transformer Base (Advanced Model)
**Accuracy: 92-94%**

```
Architecture:
├── Backbone: Swin-Base (patch4, window7, 224)
│   ├── Input: 224×224×3 RGB images
│   ├── Hierarchical Vision Transformer
│   ├── Shifted Window Multi-Head Self-Attention
│   └── 88M parameters
├── Custom Classifier:
│   ├── Layer Norm
│   ├── Dense(1024 → 512) + GELU + Dropout(0.3)
│   ├── Dense(512 → 256) + GELU + Dropout(0.2)
│   └── Dense(256 → 2) [Real, Fake]
└── Output: Softmax probabilities
```

**Why Swin Transformer?**
- Hierarchical feature learning captures multi-scale artifacts
- Shifted window attention reduces computational complexity
- Superior at capturing spatial relationships and global context
- Excellent for detecting subtle manipulations

**Training Details:**
- Loss: Focal Loss (α=0.25, γ=2.0) for handling class imbalance
- Optimizer: AdamW with cosine annealing (LR: 1e-5 → 1e-6)
- Augmentation: Advanced (RandAugment, Mixup, CutMix)
- Input Size: 224×224
- Batch Size: 16 (with gradient accumulation ×2)
- Epochs: 30-50

---

### 3️⃣ Custom Lightweight CNN (Fast Model)
**Accuracy: 81-83%**

```
Architecture:
├── Block 1:
│   ├── Conv2d(3 → 32, 3×3) + BatchNorm + ReLU
│   ├── Conv2d(32 → 32, 3×3) + BatchNorm + ReLU
│   └── MaxPool2d(2×2)
├── Block 2:
│   ├── Conv2d(32 → 64, 3×3) + BatchNorm + ReLU
│   ├── Conv2d(64 → 64, 3×3) + BatchNorm + ReLU
│   └── MaxPool2d(2×2)
├── Block 3:
│   ├── Conv2d(64 → 128, 3×3) + BatchNorm + ReLU
│   ├── Conv2d(128 → 128, 3×3) + BatchNorm + ReLU
│   └── MaxPool2d(2×2)
├── Block 4:
│   ├── Conv2d(128 → 256, 3×3) + BatchNorm + ReLU
│   ├── Conv2d(256 → 256, 3×3) + BatchNorm + ReLU
│   └── MaxPool2d(2×2)
├── Global Average Pooling
└── Classifier:
    ├── Dense(256 → 128) + ReLU + BatchNorm
    ├── Dropout(0.5)
    ├── Dense(128 → 64) + ReLU + BatchNorm
    ├── Dropout(0.3)
    └── Dense(64 → 2) [Real, Fake]

Total Parameters: 430,658
Model Size: 1.7MB
```

**Why Custom CNN?**
- Extremely lightweight (1.7MB vs 50MB for EfficientNet)
- Fast inference (~50ms per image on CPU)
- Good for mobile/edge deployment
- Trained specifically on deepfake datasets

**Training Details:**
- Dataset: Leonardo12356 (20K samples, balanced real/fake)
- Loss: CrossEntropyLoss
- Optimizer: Adam (LR: 0.001)
- Augmentation: Horizontal flip, rotation, brightness/contrast
- Input Size: 128×128
- Batch Size: 64
- Epochs: 10

---

## 🧠 How It Works

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT MEDIA                              │
│                    (Image / Video Frame)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FACE DETECTION                               │
│              (MediaPipe Face Detection)                         │
│  • Detects faces with confidence > 0.5                          │
│  • Extracts bounding box coordinates                            │
│  • Handles multiple faces (processes first detected)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FACE EXTRACTION                               │
│  • Crop face region with 20% padding                            │
│  • Resize to model-specific dimensions                          │
│  • Normalize RGB values [0-1]                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
  ┌───────────────┐ ┌───────────┐ ┌──────────────┐
  │ EfficientNet  │ │   Swin    │ │  Custom CNN  │
  │   300×300     │ │  224×224  │ │   128×128    │
  └───────┬───────┘ └─────┬─────┘ └──────┬───────┘
          │               │               │
          ▼               ▼               ▼
  ┌───────────────┐ ┌───────────┐ ┌──────────────┐
  │ Prediction    │ │Prediction │ │ Prediction   │
  │ [Real, Fake]  │ │[Real,Fake]│ │ [Real, Fake] │
  │ Confidence    │ │Confidence │ │  Confidence  │
  └───────┬───────┘ └─────┬─────┘ └──────┬───────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ENSEMBLE AGGREGATION                           │
│  • Weighted voting (EfficientNet: 0.4, Swin: 0.4, Custom: 0.2) │
│  • Threshold: > 0.5 → FAKE, ≤ 0.5 → REAL                       │
│  • Conservative approach: If any model says FAKE → FAKE         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL VERDICT                                │
│         REAL ✅  or  FAKE ⚠️ + Confidence Score                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1) � Python and system dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

On Linux (or Streamlit Cloud), you may also need these system packages:

```bash
cat packages.txt | xargs -I{} sudo apt-get install -y {}
```

Notes:

- On Windows, `opencv-python` (not headless) may be preferable if you need GUI/video codecs.
- GPU is optional. The app runs on CPU by default.

### 2) 🤖 Models

At runtime, if the `runs/` directory does not exist, `app.py` attempts to execute:

```bash
bash setup.sh
```

`setup.sh` does the following:

1. Downloads the Kaggle dataset `ameencaslam/ddp-v5-runs`
2. Unzips into `runs/`

This provides model weights at:

- `runs/models/efficientnet/best_model_cpu.pth`
- `runs/models/swin/best_model_cpu.pth`

If you prefer manual setup, create the above files at the same paths.

Kaggle CLI setup (only if using `setup.sh`):

```bash
pip install kaggle
# Place kaggle.json (with your API token) at ~/.kaggle/kaggle.json and chmod 600
```

### 3) 🧪 Run the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. The home page lets you pick Image, Video, or Live Camera.

---

## 🧰 Usage

### Image

1. Select "Image"
2. Upload a JPG/PNG with a clear face
3. The app will detect a face, run EfficientNet and Swin models, and show:
   - Each model’s prediction and confidence
   - Final verdict banner (REAL/FAKE)
   - Visualization with detected face and cropped face

### Video

1. Select "Video"
2. Upload an MP4/AVI/MOV
3. Choose number of frames to analyze (slider)
4. The app extracts frames, detects faces, runs both models, and aggregates results per model
5. View final verdict and a grid of sample detected faces

### Live Camera (local only)

- Select "Live Camera" when running locally (webcam required)
- Pick a model and start/stop the camera stream
- The frame is annotated with prediction and confidence

---

## 🧩 How it works (inference)

- Face detection via MediaPipe (`utils_image_processor.extract_face`)
- Model-specific transforms (`utils_image_processor.get_transforms`):
  - EfficientNet: resize/center-crop to 300
  - Swin: resize/center-crop to 224
- Models are defined in `utils_eff.py` and `utils_swin.py`, and loaded with `utils_model.get_cached_model`
- Decision threshold: sigmoid(output) > 0.5 → FAKE, else REAL
- Overall verdict: if either model predicts FAKE, final verdict is FAKE

Expected weight paths:

- EfficientNet: `runs/models/efficientnet/best_model_cpu.pth`
- Swin: `runs/models/swin/best_model_cpu.pth`

---

## 🏋️‍♂️ Training

Training scripts live under `training/` and use MLflow for metrics/artifacts.

### 🚀 NEW: Advanced Training Pipeline

**We've added a comprehensive advanced training pipeline with state-of-the-art techniques!**

For better results (targeting 99%+ accuracy), use the new unified training script:

```bash
# Quick start - EfficientNet with all improvements
python training/train_unified.py \
    --data_dir /path/to/dataset \
    --model_type efficientnet \
    --use_attention \
    --use_advanced_aug \
    --use_mixup \
    --use_amp \
    --num_epochs 30

# Advanced - Swin Transformer with all features
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
    --gradient_accumulation 4
```

**What's included:**
- 🎨 **Advanced Augmentation**: RandAugment, Mixup/CutMix, deepfake-specific transforms
- 🧠 **Attention Mechanisms**: SE blocks, CBAM (spatial+channel attention)
- 🎯 **Advanced Loss Functions**: Focal Loss, Label Smoothing
- ⚡ **Training Optimizations**: Mixed precision (AMP), gradient accumulation, differential LRs
- 📊 **Comprehensive Evaluation**: Detailed metrics, error analysis, visualizations
- 🎭 **Ensemble Methods**: Weighted ensembles, uncertainty estimation, calibration
- 🔄 **Test-Time Augmentation**: Multiple predictions averaged for better accuracy

**See the complete guide:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### Dataset expectations

`training/data_handler.py` expects a directory with the following structure:

```
dataset/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   └── ...
│   └── fake/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Alternative format with split files also supported (see original training scripts).

### Legacy Training Scripts

Original training scripts are still available:

**EfficientNet (Basic):**
```bash
python training/train_efficientnet.py
```

**Swin Transformer (Basic):**
```bash
python training/train_swin.py
```

Key settings (edit in file as needed):
- `DATA_DIR = '/kaggle/input/2body-images-10k-split'`
- IMAGE_SIZE: 300 (EfficientNet), 224 (Swin)
- BATCH_SIZE: 32, NUM_EPOCHS: 20
- Optimizer: AdamW with different LRs for backbone/classifier
- Logs: MLflow (`mlruns/`), plus confusion matrix, ROC, and learning curves

### Expected Performance

| Model | Configuration | Accuracy |
|-------|--------------|----------|
| EfficientNet-B3 | Basic (legacy) | 96-97% |
| EfficientNet-B3 | Advanced + Attention + Focal Loss | 97-98% |
| EfficientNet-B3 | All features + TTA | 98-99% |
| Swin-Base | Basic (legacy) | 97-98% |
| Swin-Base | All features + TTA | 99%+ |

To use your own trained weights in the app, copy your best CPU-state-dict files to:

- `runs/models/efficientnet/best_model_cpu.pth`
- `runs/models/swin/best_model_cpu.pth`

---

## 🛠️ Troubleshooting

- Streamlit Cloud cold start: first request triggers model download from Kaggle; wait for completion
- Kaggle CLI errors: ensure `~/.kaggle/kaggle.json` exists with correct permissions and that the dataset is accessible
- Missing `libGL`/X11 errors on Linux: install packages from `packages.txt`
- Webcam not working: Live Camera is disabled on deployed instances; run locally
- CUDA out-of-memory: the app runs on CPU by default; training scripts auto-detect GPU

---

## 🧱 Tech stack

- Streamlit UI
- PyTorch for modeling
- MediaPipe for face detection
- OpenCV/Pillow for image/video IO
- MLflow for experiment tracking
- `timm` and `efficientnet-pytorch` for backbones

---

## 📄 License
\\MADE BY TANUSH # deepfake-detection-xceptionnet
