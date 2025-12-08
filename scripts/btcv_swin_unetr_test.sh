#!/bin/bash
# Quick test script for Swin UNETR on AMOS22 dataset
# Single seed, fewer epochs for testing

python main.py swin_unetr \
--dataset AMOS22 \
--data-path /deeparea/sokabe/Dataset/amos22/only_CT \
--output_dir ./output/swin_unetr_amos22_test \
--num_tasks 4 \
--batch-size 1 \
--epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--weight-decay 1e-5 \
--sched cosine \
--warmup-epochs 2 \
--clip-grad 1.0 \
--img_size 96 96 96 \
--roi_size 96 96 96 \
--in_channels 1 \
--out_channels 16 \
--feature_size 48 \
--loss_type dice_ce \
--num_workers 2 \
--cache_rate 0.0 \
--val_interval 2 \
--save_interval 5 \
--print_freq 5 \
--seed 42 \
--device cuda
