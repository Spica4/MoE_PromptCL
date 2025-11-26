import argparse


def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=2, type=int, help='Batch size per device (smaller for 3D data)')
    subparsers.add_argument('--epochs', default=100, type=int)

    # Swin UNETR Model parameters
    subparsers.add_argument('--model', default='swin_unetr', type=str, metavar='MODEL',
                            help='Name of model to train')
    subparsers.add_argument('--img_size', default=(96, 96, 96), type=tuple,
                            help='Input image size (D, H, W) for 3D')
    subparsers.add_argument('--in_channels', default=1, type=int,
                            help='Number of input channels (1 for CT/MRI)')
    subparsers.add_argument('--out_channels', default=14, type=int,
                            help='Number of output segmentation classes')
    subparsers.add_argument('--feature_size', default=48, type=int,
                            help='Feature size for Swin UNETR')
    subparsers.add_argument('--drop_rate', default=0.0, type=float,
                            help='Dropout rate')
    subparsers.add_argument('--attn_drop_rate', default=0.0, type=float,
                            help='Attention dropout rate')
    subparsers.add_argument('--dropout_path_rate', default=0.0, type=float,
                            help='Drop path rate')
    subparsers.add_argument('--use_checkpoint', default=False, type=bool,
                            help='Use gradient checkpointing to save memory')
    subparsers.add_argument('--spatial_dims', default=3, type=int,
                            help='Spatial dimensions (3 for 3D)')

    # Pretrained weights
    subparsers.add_argument('--pretrained', default=False, type=bool,
                            help='Load pretrained Swin UNETR weights')
    subparsers.add_argument('--pretrained_path', default='', type=str,
                            help='Path to pretrained weights')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer (default: "adamw")')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: (0.9, 0.999))')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                            help='Clip gradient norm (default: 1.0)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=1e-5,
                            help='weight decay (default: 1e-5)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True,
                            help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                            help='LR scheduler (default: "cosine")')
    subparsers.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='learning rate (default: 1e-4)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                            help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                            help='lower lr bound for cyclic schedulers (1e-6)')
    subparsers.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                            help='epochs to warmup LR')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                            help='epochs to cooldown LR at min_lr')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                            help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True,
                            help='scaling lr by batch size (default: True)')

    # Data augmentation parameters for 3D medical images
    subparsers.add_argument('--roi_size', default=(96, 96, 96), type=tuple,
                            help='ROI size for random crop')
    subparsers.add_argument('--use_smart_cache', default=False, type=bool,
                            help='Use MONAI SmartCache for efficient data loading')
    subparsers.add_argument('--cache_rate', default=0.0, type=float,
                            help='Cache rate for dataset (0.0 = no cache, 1.0 = full cache)')
    subparsers.add_argument('--rand_flip_prob', default=0.5, type=float,
                            help='Probability of random flip augmentation')
    subparsers.add_argument('--rand_rotate_prob', default=0.5, type=float,
                            help='Probability of random rotation augmentation')
    subparsers.add_argument('--rand_scale_prob', default=0.5, type=float,
                            help='Probability of random scaling augmentation')
    subparsers.add_argument('--rand_shift_prob', default=0.5, type=float,
                            help='Probability of random shift augmentation')

    # Segmentation loss parameters
    subparsers.add_argument('--loss_type', default='dice_ce', type=str,
                            choices=['dice', 'dice_ce', 'dice_focal'],
                            help='Loss function type')
    subparsers.add_argument('--dice_loss_weight', default=0.5, type=float,
                            help='Weight for dice loss in combined loss')
    subparsers.add_argument('--ce_loss_weight', default=0.5, type=float,
                            help='Weight for cross-entropy loss in combined loss')
    subparsers.add_argument('--smooth_nr', default=0.0, type=float,
                            help='Smoothing numerator for dice loss')
    subparsers.add_argument('--smooth_dr', default=1e-5, type=float,
                            help='Smoothing denominator for dice loss')

    # Data parameters
    subparsers.add_argument('--data-path', default='/local_datasets/medical/', type=str,
                            help='dataset path')
    subparsers.add_argument('--dataset', default='BTCV', type=str,
                            help='dataset name (BTCV, BraTS, etc.)')
    subparsers.add_argument('--output_dir', default='./output/swin_unetr',
                            help='path where to save')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=4, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader')
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters for medical segmentation
    subparsers.add_argument('--num_tasks', default=5, type=int,
                            help='number of sequential tasks (e.g., different organs)')
    subparsers.add_argument('--task_inc', default=True, type=bool,
                            help='task incremental learning for multi-organ segmentation')
    subparsers.add_argument('--organ_mapping', default={
                                0: 'background',
                                1: 'spleen',
                                2: 'right_kidney',
                                3: 'left_kidney',
                                4: 'gallbladder',
                                5: 'esophagus',
                                6: 'liver',
                                7: 'stomach',
                                8: 'aorta',
                                9: 'postcava',
                                10: 'portal_vein',
                                11: 'pancreas',
                                12: 'right_adrenal',
                                13: 'left_adrenal'
                            }, type=dict, help='Organ label mapping')

    # NoRGa Prompt parameters for 3D
    subparsers.add_argument('--use_e_prompt', default=True, type=bool,
                            help='if using E-Prompt')
    subparsers.add_argument('--e_prompt_layer_idx', default=[0, 1, 2, 3], type=int, nargs="+",
                            help='the layer index of the E-Prompt in encoder')
    subparsers.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool,
                            help='if using prefix tune for E-Prompt')
    subparsers.add_argument('--larger_prompt_lr', action='store_true',
                            help='if using larger prompt lr')

    # Prompt pool parameters
    subparsers.add_argument('--prompt_pool', default=True, type=bool)
    subparsers.add_argument('--size', default=5, type=int,
                            help='prompt pool size (number of tasks)')
    subparsers.add_argument('--length', default=10, type=int,
                            help='prompt length')
    subparsers.add_argument('--top_k', default=1, type=int,
                            help='top k prompts to select')
    subparsers.add_argument('--initializer', default='uniform', type=str)
    subparsers.add_argument('--prompt_key', default=True, type=bool)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=True, type=bool)
    subparsers.add_argument('--shared_prompt_pool', default=True, type=bool)
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool)
    subparsers.add_argument('--embedding_key', default='mean', type=str,
                            help='key for prompt selection (mean pooling of features)')
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=0.5, type=float)

    # Freeze parameters
    subparsers.add_argument('--freeze', default=['encoder'], nargs='*', type=list,
                            help='freeze part in backbone model')

    # Task inference parameters
    subparsers.add_argument('--train_inference_task_only', action='store_true',
                            help='Train task inference model only')
    subparsers.add_argument('--ca_epochs', default=30, type=int,
                            help='Epochs for task inference training')
    subparsers.add_argument('--ca_lr', default=0.005, type=float,
                            help='Learning rate for task inference')
    subparsers.add_argument('--ca_storage_efficient_method', default='multi-centroid',
                            choices=['covariance', 'multi-centroid', 'variance'], type=str)
    subparsers.add_argument('--n_centroids', default=10, type=int)
    subparsers.add_argument('--prompt_momentum', default=0.01, type=float)
    subparsers.add_argument('--reg', default=0.01, type=float)

    # Non-linear gate activation (NoRGa)
    subparsers.add_argument('--gate_act', default='tanh', type=str,
                            help='gate activation function for NoRGa')
    subparsers.add_argument('--use_norga', default=True, type=bool,
                            help='Use NoRGa (non-linear residual gates)')

    # Evaluation metrics for segmentation
    subparsers.add_argument('--eval_metrics', default=['dice', 'hausdorff', 'iou'],
                            nargs='+', type=list,
                            help='Metrics to compute during evaluation')
    subparsers.add_argument('--save_predictions', default=False, type=bool,
                            help='Save prediction masks during evaluation')

    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10,
                            help='The frequency of printing')
    subparsers.add_argument('--val_interval', type=int, default=5,
                            help='Validation interval (epochs)')
    subparsers.add_argument('--save_interval', type=int, default=10,
                            help='Save checkpoint interval (epochs)')
