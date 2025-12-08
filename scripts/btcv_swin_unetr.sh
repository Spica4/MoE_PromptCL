#!/bin/bash
# Training script for Swin UNETR on AMOS22 dataset
# 4-task continual learning for organ segmentation

for seed in 42 40 44
do
python main.py swin_unetr \
--dataset AMOS22 \
--data-path /deeparea/sokabe/Dataset/amos22/only_CT \
--output_dir ./output/swin_unetr_amos22_4tasks_seed$seed \
--num_tasks 4 \
--batch-size 2 \
--epochs 100 \
--lr 1e-4 \
--min-lr 1e-6 \
--weight-decay 1e-5 \
--sched cosine \
--warmup-epochs 10 \
--clip-grad 1.0 \
--img_size 96 96 96 \
--roi_size 96 96 96 \
--in_channels 1 \
--out_channels 14 \
--feature_size 48 \
--loss_type dice_ce \
--num_workers 4 \
--cache_rate 0.0 \
--val_interval 5 \
--save_interval 10 \
--print_freq 10 \
--seed $seed \
--device cuda
done
