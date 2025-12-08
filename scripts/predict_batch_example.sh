#!/bin/bash
# Example script for running batch inference on multiple .nii.gz images

# Set your paths
INPUT_DIR="/deeparea/sokabe/Dataset/amos22/only_CT/imagesTs"  # テスト画像のディレクトリ
OUTPUT_DIR="./output/predictions/batch"  # 出力予測のディレクトリ
CHECKPOINT="./output/swin_unetr_amos22_test/best_checkpoint.pth"  # 学習済みモデル

# Run batch inference
python predict_batch.py \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--checkpoint $CHECKPOINT \
--img_size 96 96 96 \
--roi_size 96 96 96 \
--in_channels 1 \
--out_channels 16 \
--feature_size 48 \
--sw_batch_size 4 \
--overlap 0.5 \
--device cuda

echo "All predictions saved to: $OUTPUT_DIR"
