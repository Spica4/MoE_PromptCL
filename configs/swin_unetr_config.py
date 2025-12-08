"""
Configuration for Swin UNETR on 3D medical image segmentation
"""

import argparse


def get_args_parser(subparsers):
    """
    Get argument parser for Swin UNETR configuration

    Args:
        subparsers: Parser object from main argument parser
    """
    # Dataset parameters
    subparsers.add_argument('--dataset', default='BTCV', type=str,
                            help='Dataset name')
    subparsers.add_argument('--data-path', default='./local_datasets/medical/BTCV', type=str,
                            help='Path to dataset')
    subparsers.add_argument('--output_dir', default='./output/swin_unetr',
                            help='Path to save outputs')

    # Model parameters
    subparsers.add_argument('--img_size', default=[96, 96, 96], type=int, nargs=3,
                            help='Input image size (D H W)')
    subparsers.add_argument('--in_channels', default=1, type=int,
                            help='Number of input channels (1 for CT/MRI)')
    subparsers.add_argument('--out_channels', default=16, type=int,
                            help='Number of output classes (15 organs + background for AMOS22)')
    subparsers.add_argument('--feature_size', default=48, type=int,
                            help='Feature size for Swin transformer')
    subparsers.add_argument('--roi_size', default=[96, 96, 96], type=int, nargs=3,
                            help='ROI size for random crop')
    subparsers.add_argument('--spatial_dims', default=3, type=int,
                            help='Spatial dimensions (3 for 3D)')

    # MoE (Mixture of Experts) parameters
    subparsers.add_argument('--use_moe', action='store_true',
                            help='Use Mixture of Experts')
    subparsers.add_argument('--num_experts_per_task', default=4, type=int,
                            help='Number of experts per task for MoE')
    subparsers.add_argument('--prompt_length', default=5, type=int,
                            help='Length of expert prompts')
    subparsers.add_argument('--use_nonlinear_gate', action='store_true', default=True,
                            help='Use non-linear gating in MoE')
    subparsers.add_argument('--use_residual_gate', action='store_true', default=True,
                            help='Use residual connection in MoE gating (NoRGa)')

    # Training parameters
    subparsers.add_argument('--batch-size', default=2, type=int,
                            help='Batch size per GPU')
    subparsers.add_argument('--epochs', default=100, type=int,
                            help='Number of epochs')
    subparsers.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='Learning rate')
    subparsers.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                            help='Lower lr bound for cyclic schedulers')
    subparsers.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                            help='Epochs to warmup LR')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer')
    subparsers.add_argument('--weight-decay', type=float, default=1e-5,
                            help='Weight decay')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                            help='Clip gradient norm (0 = no clipping)')

    # Learning rate scheduler
    subparsers.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                            help='LR scheduler (cosine, step)')

    # Loss function
    subparsers.add_argument('--loss_type', default='dice_ce', type=str,
                            help='Loss function type (dice, dice_ce)')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=4, type=int,
                            help='Number of sequential tasks (each task = one organ)')

    # Data loading
    subparsers.add_argument('--num_workers', default=4, type=int,
                            help='Number of data loading workers')
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader')
    subparsers.add_argument('--cache_rate', default=0.0, type=float,
                            help='Cache rate for MONAI CacheDataset (0.0-1.0)')

    # Misc
    subparsers.add_argument('--print_freq', default=10, type=int,
                            help='Print frequency')
    subparsers.add_argument('--val_interval', default=5, type=int,
                            help='Validation interval (epochs)')
    subparsers.add_argument('--save_interval', default=10, type=int,
                            help='Save checkpoint interval (epochs)')
    subparsers.add_argument('--device', default='cuda',
                            help='Device to use for training')
    subparsers.add_argument('--seed', default=42, type=int,
                            help='Random seed')
    subparsers.add_argument('--eval', action='store_true',
                            help='Perform evaluation only')

    # Pretrained model
    subparsers.add_argument('--pretrained', action='store_true',
                            help='Use pretrained model')
    subparsers.add_argument('--pretrained_path', default='', type=str,
                            help='Path to pretrained model')

    # Distributed training
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='Number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://',
                            help='URL used to set up distributed training')
