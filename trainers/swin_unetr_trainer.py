"""
Trainer for Swin UNETR segmentation with continual learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import datetime
import os
from pathlib import Path

from vits.swin_unetr_model import create_swin_unetr_model
from continual_datasets.medical_segmentation import (
    build_continual_medical_dataloader,
    create_task_split_for_organs,
)
from engines.swin_unetr_engine import (
    train_one_epoch,
    evaluate,
    create_segmentation_loss,
)
import utils


def train(args):
    """
    Main training function for Swin UNETR continual learning

    Args:
        args: Configuration arguments
    """
    device = torch.device(args.device)

    # Create task splits for continual learning (cumulative)
    total_organs = 15  # AMOS22の全臓器数

    task_organ_lists = create_task_split_for_organs(
        num_tasks=args.num_tasks,
        total_organs=total_organs,
        cumulative=True  # 累積学習モード
    )

    print(f"Continual learning task splits (cumulative):")
    for i, organs in enumerate(task_organ_lists):
        num_classes = len(organs) + 1  # organs + background
        print(f"  Task {i}: Organs {organs} -> {num_classes} classes (background + {len(organs)} organs)")

    # Build data loaders for all tasks
    data_loaders = []
    for task_id in range(args.num_tasks):
        train_loader = build_continual_medical_dataloader(
            dataset_name=args.dataset,
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            task_id=task_id,
            organ_list=task_organ_lists[task_id],
            mode='train',
            roi_size=args.roi_size,
            cache_rate=args.cache_rate,
        )

        val_loader = build_continual_medical_dataloader(
            dataset_name=args.dataset,
            data_dir=args.data_path,
            batch_size=1,
            num_workers=args.num_workers,
            task_id=task_id,
            organ_list=task_organ_lists[task_id],
            mode='val',
            roi_size=args.roi_size,
            cache_rate=0.0,
        )

        data_loaders.append({
            'train': train_loader,
            'val': val_loader,
        })

    print(f"Data loaders created for {args.num_tasks} tasks")

    # Create model with initial output channels for Task 0
    initial_out_channels = len(task_organ_lists[0]) + 1  # background + first organ
    print(f"Creating Swin UNETR model with initial {initial_out_channels} output channels")

    # Temporarily override out_channels for model creation
    original_out_channels = args.out_channels
    args.out_channels = initial_out_channels
    model = create_swin_unetr_model(args)
    args.out_channels = original_out_channels  # Restore original value

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters:,}')

    # Evaluation mode
    if args.eval:
        print("Evaluation mode - not implemented yet")
        return

    # Create loss function
    criterion = create_segmentation_loss(args)
    print(f"Using loss function: {criterion}")

    # Training loop for continual learning
    print(f"Start continual learning training for {args.num_tasks} tasks")
    start_time = time.time()

    for task_id in range(args.num_tasks):
        print(f"\n{'=' * 80}")
        print(f"Training Task {task_id}: Organs {task_organ_lists[task_id]}")
        print(f"{'=' * 80}\n")

        # Expand model output layer if needed (for tasks after Task 0)
        if task_id > 0:
            new_out_channels = len(task_organ_lists[task_id]) + 1  # background + current organs
            current_out_channels = model.out_channels
            if new_out_channels > current_out_channels:
                print(f"Expanding model from {current_out_channels} to {new_out_channels} output channels")
                model.expand_output_layer(new_out_channels)
                # Move model back to device after expansion
                model.to(device)
                print(f"Model expansion complete")

        train_loader = data_loaders[task_id]['train']
        val_loader = data_loaders[task_id]['val']

        # Create optimizer for current task
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Create learning rate scheduler
        if args.sched == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )
        elif args.sched == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            lr_scheduler = None

        # Train for current task
        best_dice = 0.0
        for epoch in range(args.epochs):
            # Training
            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_grad,
                task_id=task_id,
                organ_list=task_organ_lists[task_id],
                args=args,
            )

            # Validation
            if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
                val_stats = evaluate(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    task_id=task_id,
                    organ_list=task_organ_lists[task_id],
                    args=args,
                )

                print(f"Epoch {epoch + 1}/{args.epochs} - Val Dice: {val_stats['dice']:.4f}")

                # Save best checkpoint
                if val_stats['dice'] > best_dice:
                    best_dice = val_stats['dice']
                    save_checkpoint(
                        model=model,
                        task_id=task_id,
                        epoch=epoch,
                        args=args,
                        filename='best_checkpoint.pth'
                    )

            # Update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Save checkpoint at intervals
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model=model,
                    task_id=task_id,
                    epoch=epoch,
                    args=args,
                )

        # Save final checkpoint for this task
        save_checkpoint(
            model=model,
            task_id=task_id,
            epoch=args.epochs - 1,
            args=args,
            filename=f'task{task_id}_checkpoint.pth'
        )

        print(f"\nTask {task_id} training completed. Best Dice: {best_dice:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTotal training time: {total_time_str}")


def save_checkpoint(model, task_id, epoch, args, filename=None):
    """
    Save model checkpoint

    Args:
        model: Model to save
        task_id: Current task ID
        epoch: Current epoch
        args: Arguments
        filename: Optional filename
    """
    checkpoint_dir = Path(args.output_dir) / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f'task{task_id}_epoch{epoch}.pth'

    checkpoint_path = checkpoint_dir / filename

    torch.save({
        'model': model.state_dict(),
        'task_id': task_id,
        'epoch': epoch,
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")
