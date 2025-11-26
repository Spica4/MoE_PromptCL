#!/bin/bash
# Download pretrained Swin UNETR models from MONAI

set -e  # Exit on error

echo "=========================================="
echo "Downloading Swin UNETR Pretrained Models"
echo "=========================================="
echo

# Create directory
CHECKPOINT_DIR="./checkpoints/swin_unetr"
mkdir -p ${CHECKPOINT_DIR}

echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo

# Option 1: Self-supervised pretrained model (recommended for continual learning)
echo "1. Downloading self-supervised pretrained Swin UNETR..."
echo "   Pretrained on 5,050 CT images with self-supervised learning"
echo

if [ -f "${CHECKPOINT_DIR}/model_swinvit.pt" ]; then
    echo "   ✓ model_swinvit.pt already exists, skipping..."
else
    wget -c https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt \
        -O ${CHECKPOINT_DIR}/model_swinvit.pt
    echo "   ✓ Downloaded model_swinvit.pt"
fi
echo

# Option 2: BTCV fine-tuned model (for reference)
echo "2. Downloading BTCV fine-tuned Swin UNETR..."
echo "   Fine-tuned on BTCV multi-organ segmentation dataset"
echo

if [ -f "${CHECKPOINT_DIR}/swin_unetr_btcv.pt" ]; then
    echo "   ✓ swin_unetr_btcv.pt already exists, skipping..."
else
    wget -c https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.btcv_f48_r.pt \
        -O ${CHECKPOINT_DIR}/swin_unetr_btcv.pt
    echo "   ✓ Downloaded swin_unetr_btcv.pt"
fi
echo

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo
echo "Available models:"
ls -lh ${CHECKPOINT_DIR}
echo
echo "Usage:"
echo "  # Use self-supervised pretrained (recommended for continual learning)"
echo "  python main.py swin_unetr_norgaprompt \\"
echo "      --pretrained True \\"
echo "      --pretrained_path ./checkpoints/swin_unetr/model_swinvit.pt \\"
echo "      ..."
echo
echo "  # Use BTCV fine-tuned model"
echo "  python main.py swin_unetr_norgaprompt \\"
echo "      --pretrained True \\"
echo "      --pretrained_path ./checkpoints/swin_unetr/swin_unetr_btcv.pt \\"
echo "      ..."
echo
