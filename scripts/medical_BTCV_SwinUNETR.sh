#!/bin/bash
# Training script for Swin UNETR on BTCV Multi-Organ Segmentation
# with NoRGa prompts for continual learning

# Default configuration
DATASET="BTCV"
DATA_PATH="./local_datasets/medical/BTCV"
OUTPUT_DIR="./output/swin_unetr_btcv"
BATCH_SIZE=2
EPOCHS=100
LR=1e-4
NUM_TASKS=5
SEED=42

# Parse command line arguments (optional)
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Swin UNETR Training on ${DATASET}"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Num tasks: ${NUM_TASKS}"
echo "Seed: ${SEED}"
echo "=========================================="
echo

# Run training
python main.py swin_unetr_norgaprompt \
    --dataset ${DATASET} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_seed${SEED} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --seed ${SEED} \
    --num_tasks ${NUM_TASKS} \
    --pretrained True \
    --pretrained_path ./checkpoints/swin_unetr/model_swinvit.pt \
    --img_size 96 96 96 \
    --in_channels 1 \
    --out_channels 14 \
    --feature_size 48 \
    --use_e_prompt True \
    --e_prompt_layer_idx 0 1 2 3 \
    --use_prefix_tune_for_e_prompt True \
    --prompt_pool True \
    --size ${NUM_TASKS} \
    --length 10 \
    --top_k 1 \
    --batchwise_prompt True \
    --prompt_key_init uniform \
    --embedding_key mean \
    --pull_constraint True \
    --pull_constraint_coeff 0.5 \
    --use_norga True \
    --gate_act tanh \
    --larger_prompt_lr \
    --sched cosine \
    --warmup-epochs 10 \
    --min-lr 1e-6 \
    --opt adamw \
    --weight-decay 1e-5 \
    --clip-grad 1.0 \
    --loss_type dice_ce \
    --dice_loss_weight 0.5 \
    --ce_loss_weight 0.5 \
    --roi_size 96 96 96 \
    --cache_rate 0.0 \
    --use_checkpoint False \
    --spatial_dims 3 \
    --eval_metrics dice hausdorff iou \
    --val_interval 5 \
    --save_interval 10 \
    --print_freq 10 \
    --num_workers 4 \
    --pin-mem \
    --device cuda \
    --task_inc True

echo
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}_seed${SEED}"
