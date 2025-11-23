#!/bin/bash

cat << "EOF"
==========================================
  MOBILENETV2 TRAINING - LIGHTWEIGHT & POWERFUL
==========================================

🏆 MobileNetV2 Architecture:
   • Google's efficient deep learning model
   • Only 3.5M parameters (vs 22M XceptionNet)
   • Pre-trained on ImageNet (transfer learning)
   • Proven 85-92% accuracy on deepfakes
   • Used in production mobile apps

⚙️  Configuration:
   - 15 epochs with CosineAnnealing LR
   - 20,000 samples (10k real + 10k fake)
   - Batch size: 24 (memory optimized)
   - Image size: 224x224
   - ImageNet pre-trained weights
   - Ultra memory efficient

🎯 Expected Accuracy: 85-92%
⏱️  Estimated Time: 50-60 minutes

✅ Why MobileNetV2?
   • 85% smaller than XceptionNet
   • Fits easily in memory
   • Transfer learning from ImageNet
   • Fast training & inference
   • Industry-proven architecture

EOF

read -p "Start training MobileNetV2? (y/n): " choice
if [ "$choice" != "y" ]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
python3 training/train_mobilenet.py
