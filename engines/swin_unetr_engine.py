"""
Training and evaluation engine for Swin UNETR segmentation
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import utils


def train_one_epoch(
    model: nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 1.0,
    task_id=-1,
    organ_list=None,
    args=None,
):
    """
    Train for one epoch

    Args:
        model: Swin UNETR model
        criterion: Loss function
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        max_norm: Gradient clipping max norm
        task_id: Current task ID
        organ_list: List of organ labels
        args: Additional arguments
    """
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1}] Task[{task_id}]'

    for batch_data in metric_logger.log_every(data_loader, args.print_freq, header):
        inputs = batch_data['image'].to(device, non_blocking=True)
        labels = batch_data['label'].to(device, non_blocking=True)

        # Forward pass
        outputs, prompt_losses = model(inputs, task_id=task_id, train=True)

        # Compute loss
        seg_loss = criterion(outputs, labels)
        total_loss = seg_loss

        # Add prompt-related losses if any
        if prompt_losses:
            for key, value in prompt_losses.items():
                if value is not None:
                    total_loss = total_loss + value

        # Check for NaN/Inf
        if not math.isfinite(total_loss.item()):
            print(f"Loss is {total_loss.item()}, stopping training")
            sys.exit(1)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Update metrics
        torch.cuda.synchronize()
        metric_logger.update(Loss=total_loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    task_id=-1,
    organ_list=None,
    args=None,
):
    """
    Evaluate model

    Args:
        model: Swin UNETR model
        data_loader: Validation data loader
        device: Device
        task_id: Current task ID
        organ_list: List of organ labels
        args: Additional arguments

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # Convert roi_size to tuple
    roi_size = tuple(args.roi_size) if isinstance(args.roi_size, list) else args.roi_size

    # Initialize MONAI metrics
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: Task[{task_id}]'

    for batch_data in metric_logger.log_every(data_loader, args.print_freq, header):
        inputs = batch_data['image'].to(device)
        labels = batch_data['label'].to(device)

        # Use sliding window inference for large 3D volumes
        outputs = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=lambda x: model(x, task_id=task_id, train=False)[0],
            overlap=0.5,
        )

        # Convert to one-hot for metric computation
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        preds_onehot = F.one_hot(preds.squeeze(1), num_classes=args.out_channels)
        preds_onehot = preds_onehot.permute(0, 4, 1, 2, 3).float()

        # Compute Dice metric
        dice_metric(y_pred=preds_onehot, y=labels)

    # Aggregate metrics
    dice_scores = dice_metric.aggregate()
    mean_dice = dice_scores.mean().item()

    print(f"Task {task_id} - Mean Dice Score: {mean_dice:.4f}")

    dice_metric.reset()

    results = {
        'dice': mean_dice,
    }

    return results


def create_segmentation_loss(args):
    """
    Create segmentation loss function

    Args:
        args: Configuration arguments

    Returns:
        Loss function
    """
    loss_type = getattr(args, 'loss_type', 'dice_ce')

    if loss_type == 'dice':
        criterion = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
    elif loss_type == 'dice_ce':
        criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return criterion
