#!/bin/bash
# Example script for running inference on a single .nii.gz image

# Set your paths
INPUT_IMAGE="/deeparea/sokabe/Dataset/amos22/only_CT/imagesTs/amos_0500.nii.gz"  # 例: テスト画像
OUTPUT_PRED="./output/predictions/amos_0500_pred.nii.gz"  # 出力予測
CHECKPOINT="./output/swin_unetr_amos22_test/best_checkpoint.pth"  # 学習済みモデル

# Run inference
python predict_medical.py \
--input $INPUT_IMAGE \
--output $OUTPUT_PRED \
--checkpoint $CHECKPOINT \
--img_size 96 96 96 \
--roi_size 96 96 96 \
--in_channels 1 \
--out_channels 16 \
--feature_size 48 \
--sw_batch_size 4 \
--overlap 0.5 \
--device cuda

echo "Prediction saved to: $OUTPUT_PRED"
