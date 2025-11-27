"""
Training and evaluation engine for Swin UNETR segmentation with continual learning
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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
    Train one epoch for 3D medical image segmentation

    Args:
        model: Swin UNETR model
        criterion: Loss function (DiceCE, etc.)
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        max_norm: Gradient clipping max norm
        task_id: Current task ID
        organ_list: List of organ labels for current task
        args: Additional arguments

    Returns:
        Dictionary of training metrics
    """
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('DiceLoss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}] Task[{task_id}]'

    for batch_data in metric_logger.log_every(data_loader, args.print_freq, header):
        # Get inputs and labels
        inputs = batch_data['image'].to(device, non_blocking=True)
        labels = batch_data['label'].to(device, non_blocking=True)

        # Forward pass
        outputs, prompt_losses = model(
            inputs,
            task_id=task_id,
            train=True
        )

        # Compute segmentation loss
        seg_loss = criterion(outputs, labels)

        # Add prompt-related losses
        total_loss = seg_loss
        if args.pull_constraint and 'reduce_sim' in prompt_losses and prompt_losses['reduce_sim'] is not None:
            total_loss = total_loss + args.pull_constraint_coeff * prompt_losses['reduce_sim']

        # Compute Dice score for monitoring
        with torch.no_grad():
            dice_score = compute_dice_score(outputs, labels, args.out_channels)

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
        metric_logger.update(DiceLoss=seg_loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Dice'].update(dice_score.item(), n=inputs.shape[0])

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

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
    Evaluate model on validation/test data

    Args:
        model: Swin UNETR model
        data_loader: Validation data loader
        device: Device
        task_id: Current task ID
        organ_list: List of organ labels for current task
        args: Additional arguments

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # Convert roi_size to tuple if necessary (for MONAI functions)
    roi_size = tuple(args.roi_size) if isinstance(args.roi_size, list) else args.roi_size

    # Initialize MONAI metrics
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )

    if 'hausdorff' in args.eval_metrics:
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95.0,
            reduction="mean"
        )
    else:
        hausdorff_metric = None

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: Task[{task_id}]'

    with torch.no_grad():
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

            # Convert to predictions
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            # Compute Dice metric
            dice_metric(y_pred=preds, y=labels)

            # Compute Hausdorff distance if requested
            if hausdorff_metric is not None:
                try:
                    hausdorff_metric(y_pred=preds, y=labels)
                except:
                    pass  # Hausdorff may fail for some cases

    # Aggregate metrics
    dice_scores = dice_metric.aggregate()
    mean_dice = dice_scores.mean().item()

    print(f"Task {task_id} - Mean Dice Score: {mean_dice:.4f}")

    if hausdorff_metric is not None:
        hausdorff_scores = hausdorff_metric.aggregate()
        mean_hausdorff = hausdorff_scores.mean().item()
        print(f"Task {task_id} - Mean Hausdorff Distance (95%): {mean_hausdorff:.4f}")
    else:
        mean_hausdorff = 0.0

    # Reset metrics
    dice_metric.reset()
    if hausdorff_metric is not None:
        hausdorff_metric.reset()

    results = {
        'dice': mean_dice,
        'hausdorff': mean_hausdorff,
    }

    # Note: Individual organ scores are not available with reduction="mean"
    # If per-organ metrics are needed, change DiceMetric reduction to "mean_batch"

    return results


@torch.no_grad()
def evaluate_till_now(
    model: nn.Module,
    data_loaders: list,
    device: torch.device,
    task_id=-1,
    class_mask=None,
    organ_lists=None,
    args=None,
):
    """
    Evaluate on all tasks seen so far

    Args:
        model: Model to evaluate
        data_loaders: List of data loaders for each task
        device: Device
        task_id: Current task ID
        class_mask: Class mask (unused for segmentation)
        organ_lists: List of organ lists for each task
        args: Additional arguments

    Returns:
        Statistics dictionary
    """
    stat_matrix = np.zeros((3, task_id + 1))  # [rows: avg, task_avg, forgetting]

    for i in range(task_id + 1):
        test_loader = data_loaders[i]['val']
        organ_list = organ_lists[i] if organ_lists is not None else None

        results = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            task_id=i,
            organ_list=organ_list,
            args=args,
        )

        stat_matrix[0, i] = results['dice']

    # Compute average metrics
    avg_stat = stat_matrix[0, :task_id + 1].mean()
    print(f"Average Dice Score till Task {task_id}: {avg_stat:.4f}")

    # Compute per-task Dice scores
    for i in range(task_id + 1):
        print(f"Task {i} Dice Score: {stat_matrix[0, i]:.4f}")

    return stat_matrix, avg_stat


def compute_dice_score(pred_logits, target, num_classes):
    """
    Compute Dice score from prediction logits and target labels

    Args:
        pred_logits: Predicted logits (B, C, D, H, W)
        target: Target labels (B, 1, D, H, W) or (B, D, H, W)
        num_classes: Number of classes

    Returns:
        Mean Dice score across all classes (excluding background)
    """
    # Convert logits to predictions
    pred = torch.argmax(pred_logits, dim=1, keepdim=True)  # (B, 1, D, H, W)

    if len(target.shape) == 4:
        target = target.unsqueeze(1)  # (B, 1, D, H, W)

    # One-hot encode
    pred_one_hot = F.one_hot(pred.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    # Compute Dice for each class (excluding background)
    dice_scores = []
    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = pred_one_hot[:, c]
        target_c = target_one_hot[:, c]

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-5)
            dice_scores.append(dice)

    if len(dice_scores) > 0:
        return torch.stack(dice_scores).mean()
    else:
        return torch.tensor(0.0, device=pred_logits.device)


def create_segmentation_loss(args):
    """
    Create segmentation loss function based on config

    Args:
        args: Configuration arguments

    Returns:
        Loss function
    """
    if args.loss_type == 'dice':
        criterion = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr,
        )
    elif args.loss_type == 'dice_ce':
        criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=args.dice_loss_weight,
            lambda_ce=args.ce_loss_weight,
        )
    elif args.loss_type == 'dice_focal':
        criterion = DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=args.dice_loss_weight,
            lambda_focal=1.0 - args.dice_loss_weight,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    return criterion


@torch.no_grad()
def inference_and_save(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    save_dir: str,
    task_id=-1,
    args=None,
):
    """
    Perform inference on test data and save predictions as .nii.gz files

    Args:
        model: Swin UNETR model
        data_loader: Test data loader
        device: Device
        save_dir: Directory to save predictions
        task_id: Current task ID
        args: Additional arguments

    Returns:
        List of saved file paths
    """
    import nibabel as nib
    from pathlib import Path

    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert roi_size to tuple if necessary (for MONAI functions)
    roi_size = tuple(args.roi_size) if isinstance(args.roi_size, list) else args.roi_size

    saved_files = []

    print(f"Starting inference for Task {task_id}...")
    for idx, batch_data in enumerate(data_loader):
        inputs = batch_data['image'].to(device)

        # Get original image path for naming
        if 'image_meta_dict' in batch_data and 'filename_or_obj' in batch_data['image_meta_dict']:
            orig_path = batch_data['image_meta_dict']['filename_or_obj'][0]
            filename = Path(orig_path).stem.replace('.nii', '')  # Remove .nii from .nii.gz
        else:
            filename = f"pred_{idx:04d}"

        # Use sliding window inference for large 3D volumes
        outputs = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=lambda x: model(x, task_id=task_id, train=False)[0],
            overlap=0.5,
        )

        # Get predictions (argmax over channel dimension)
        preds = torch.argmax(outputs, dim=1, keepdim=True)  # (B, 1, D, H, W)

        # Save each prediction in the batch
        for b in range(preds.shape[0]):
            pred_np = preds[b, 0].cpu().numpy().astype(np.uint8)  # (D, H, W)

            # Create NIfTI image
            nii_img = nib.Nifti1Image(pred_np, affine=np.eye(4))

            # Save file
            save_path = save_dir / f"{filename}_task{task_id}.nii.gz"
            nib.save(nii_img, str(save_path))
            saved_files.append(str(save_path))

            if idx % 10 == 0:
                print(f"Saved prediction: {save_path}")

    print(f"Inference complete. Saved {len(saved_files)} predictions to {save_dir}")
    return saved_files


@torch.no_grad()
def evaluate_test_predictions(
    pred_dir: str,
    label_dir: str,
    out_channels: int,
    save_excel: str = None,
):
    """
    Evaluate test predictions against ground truth labels and save to Excel

    Args:
        pred_dir: Directory containing prediction .nii.gz files
        label_dir: Directory containing ground truth .nii.gz files
        out_channels: Number of output channels (classes)
        save_excel: Path to save Excel file with results

    Returns:
        Dictionary of evaluation metrics
    """
    import nibabel as nib
    from pathlib import Path
    import pandas as pd

    pred_dir = Path(pred_dir)
    label_dir = Path(label_dir)

    # Initialize MONAI metrics
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",  # Keep per-class scores
        get_not_nans=False
    )

    results_list = []

    print(f"Evaluating predictions in {pred_dir} against labels in {label_dir}...")

    for pred_path in sorted(pred_dir.glob('*.nii.gz')):
        # Find corresponding label file
        # Assuming prediction is named like "img0001_task0.nii.gz"
        # and label is "label0001.nii.gz"
        base_name = pred_path.stem.replace('.nii', '')  # Remove .nii from .nii.gz
        # Extract original image name (remove _taskX suffix)
        orig_name = '_'.join(base_name.split('_')[:-1]) if '_task' in base_name else base_name
        label_name = orig_name.replace('img', 'label') + '.nii.gz'
        label_path = label_dir / label_name

        if not label_path.exists():
            print(f"Warning: Label not found for {pred_path.name}, skipping...")
            continue

        # Load prediction and label
        pred_nii = nib.load(pred_path)
        label_nii = nib.load(label_path)

        pred_np = pred_nii.get_fdata()
        label_np = label_nii.get_fdata()

        # Convert to torch tensors
        pred_tensor = torch.from_numpy(pred_np).long().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        label_tensor = torch.from_numpy(label_np).long().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Convert prediction to one-hot
        pred_onehot = F.one_hot(pred_tensor.squeeze(1), num_classes=out_channels)  # (1, D, H, W, C)
        pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()  # (1, C, D, H, W)

        # Compute Dice score
        dice_metric(y_pred=pred_onehot, y=label_tensor)

    # Aggregate metrics
    dice_scores = dice_metric.aggregate()  # (C-1,) tensor, excluding background
    mean_dice = dice_scores.mean().item()

    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Per-class Dice Scores:")
    for i, score in enumerate(dice_scores):
        print(f"  Class {i+1}: {score.item():.4f}")

    # Create results dictionary
    results = {
        'mean_dice': mean_dice,
    }

    # Add per-class scores
    for i, score in enumerate(dice_scores):
        results[f'dice_class_{i+1}'] = score.item()

    # Save to Excel if path provided
    if save_excel:
        df_data = {
            'Metric': ['Mean Dice'] + [f'Dice Class {i+1}' for i in range(len(dice_scores))],
            'Score': [mean_dice] + [score.item() for score in dice_scores]
        }
        df = pd.DataFrame(df_data)
        df.to_excel(save_excel, index=False)
        print(f"Results saved to {save_excel}")

    return results


if __name__ == "__main__":
    # Test loss creation
    import argparse

    args = argparse.Namespace(
        loss_type='dice_ce',
        dice_loss_weight=0.5,
        ce_loss_weight=0.5,
        smooth_nr=0.0,
        smooth_dr=1e-5,
    )

    criterion = create_segmentation_loss(args)
    print(f"Created loss: {criterion}")

    # Test Dice computation
    B, C, D, H, W = 2, 14, 96, 96, 96
    pred_logits = torch.randn(B, C, D, H, W)
    target = torch.randint(0, C, (B, 1, D, H, W))

    dice = compute_dice_score(pred_logits, target, C)
    print(f"Dice score: {dice.item():.4f}")
