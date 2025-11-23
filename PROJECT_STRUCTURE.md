# Project Structure

This document describes the organized structure of the Deepfake Detection Project.

## 📁 Directory Structure

```
deepfake-detection-project-v5/
├── app.py                          # Main Streamlit application
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages
├── LICENSE                         # Project license
│
├── utils/                          # 🔧 Utility modules (organized package)
│   ├── __init__.py                # Package initialization
│   ├── utils_model.py             # Model loading and caching
│   ├── utils_image_input.py       # Image upload and processing
│   ├── utils_image_processor.py   # Image preprocessing and face detection
│   ├── utils_video_input.py       # Video upload and processing
│   ├── utils_video_processor.py   # Video frame extraction and analysis
│   ├── utils_live_cam.py          # Live camera detection
│   ├── utils_improved_predictor.py # Enhanced prediction with ensemble
│   ├── utils_full_frame_analysis.py # Full frame deepfake analysis
│   ├── utils_report_generator.py  # HTML report generation
│   ├── utils_format.py            # Output formatting utilities
│   ├── utils_session.py           # Session state management
│   ├── utils_eff.py               # EfficientNet model definition
│   └── utils_swin.py              # Swin Transformer model definition
│
├── training/                       # 🎓 Model training scripts
│   ├── train_efficientnet.py     # EfficientNet training
│   ├── train_swin.py              # Swin Transformer training
│   ├── train_xception.py          # XceptionNet training
│   ├── xception_model.py          # XceptionNet architecture
│   ├── data_handler.py            # Dataset loading and preprocessing
│   ├── enhanced_models.py         # Enhanced model architectures
│   └── evaluation_tools.py        # Model evaluation utilities
│
├── runs/                           # 💾 Trained model weights
│   └── models/
│       ├── efficientnet/          # EfficientNet model files
│       ├── swin/                  # Swin Transformer model files
│       └── xception_hf/           # XceptionNet model files
│
├── docs/                           # 📚 Documentation
│   ├── PROJECT_GUIDE.md           # Comprehensive project guide
│   ├── TRAINING_GUIDE.md          # Model training instructions
│   ├── QUICK_REFERENCE.md         # Quick reference guide
│   ├── CLEANUP_SUMMARY.md         # Project cleanup documentation
│   ├── FINAL_TRAINING_STATUS.md   # Training status report
│   └── XCEPTIONNET_QUICK_REFERENCE.md # XceptionNet specific guide
│
├── logs/                           # 📝 Training and execution logs
│   ├── xception_training.log      # XceptionNet training logs
│   └── xception_training_full.log # Full XceptionNet training logs
│
└── scripts/                        # 🚀 Utility scripts
    └── setup.sh                   # Initial setup script
```

## 🔍 Key Components

### Main Application
- **app.py**: The main Streamlit web application that provides the user interface for deepfake detection

### Utils Package
All utility functions are now organized in the `utils/` package with proper Python package structure:
- Model loading and inference
- Image and video processing
- Face detection and extraction
- Ensemble prediction
- Report generation
- Session management

### Training Module
Contains all model training scripts and related utilities:
- Individual model training scripts
- Data handling and preprocessing
- Model architecture definitions
- Evaluation tools

### Model Storage
- **runs/models/**: Stores trained model weights for different architectures
  - EfficientNet-B3 (pre-trained)
  - Swin Transformer (pre-trained)
  - XceptionNet (optional)

### Documentation
All project documentation is organized in the `docs/` directory for easy reference.

### Logs
Training logs and execution logs are stored in the `logs/` directory.

## 🎯 Import Structure

The project now uses proper Python package imports:

```python
# In app.py
from utils.utils_image_input import process_image_input
from utils.utils_video_input import process_video_input
from utils.utils_live_cam import show_live_camera_page

# Within utils package (relative imports)
from .utils_model import get_cached_model
from .utils_image_processor import extract_face, process_image
```

## 🚀 Benefits of This Structure

1. **Organization**: Clear separation of concerns with dedicated directories
2. **Maintainability**: Easy to find and update specific components
3. **Scalability**: Simple to add new features or models
4. **Professional**: Industry-standard Python package structure
5. **Clean**: All utility code in one package, documentation in another

## 📦 Deployment

For deployment, only the following are essential:
- `app.py`
- `utils/` (entire package)
- `runs/models/` (trained model weights)
- `requirements.txt`
- `packages.txt`
- `README.md`

The `training/`, `docs/`, `logs/`, and `scripts/` directories are for development and reference.
