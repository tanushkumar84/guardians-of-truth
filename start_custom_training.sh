#!/bin/bash

echo "=========================================="
echo "  CUSTOM CNN TRAINING - FIXED VERSION"
echo "=========================================="
echo ""
echo "⚙️  Configuration:"
echo "   - 8 epochs with early stopping"
echo "   - 50,000 samples (diverse dataset)"
echo "   - Batch size: 32"
echo "   - Image size: 128x128"
echo "   - Simpler model (~400K params)"
echo "   - Strong data augmentation"
echo "   - High dropout (50%)"
echo "   - Checkpoint auto-resume"
echo ""
echo "🎯 Target: ~85-90% (realistic accuracy)"
echo "⏱️  Estimated time: 50-60 minutes"
echo ""
echo "✅ Fixed overfitting issues:"
echo "   • Simpler model architecture"
echo "   • Much larger dataset (50k)"
echo "   • Strong regularization"
echo "   • Early stopping (patience=3)"
echo ""

read -p "Start training? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python training/train_custom_quick.py
else
    echo "Training cancelled."
fi
