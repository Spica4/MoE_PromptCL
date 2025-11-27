#!/bin/bash
# Quick test script for Swin UNETR on BTCV
# Uses smaller settings for faster testing

echo "Quick test: Swin UNETR on BTCV"
echo "================================"
echo

python main.py swin_unetr_norgaprompt \
    --dataset BTCV \
    --data-path ./local_datasets/medical/BTCV \
    --output_dir ./output/swin_unetr_btcv_test \
    --batch-size 1 \
    --epochs 5 \
    --lr 1e-4 \
    --seed 42 \
    --num_tasks 4 \
    --img_size 96 96 96 \
    --in_channels 1 \
    --out_channels 14 \
    --feature_size 48 \
    --prompt_pool True \
    --size 3 \
    --length 5 \
    --top_k 1 \
    --use_norga True \
    --gate_act tanh \
    --sched cosine \
    --loss_type dice_ce \
    --val_interval 2 \
    --save_interval 5 \
    --print_freq 5 \
    --num_workers 2 \
    --device cuda

echo
echo "Quick test completed!"
