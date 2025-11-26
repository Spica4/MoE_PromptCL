"""
Trainer for Swin UNETR segmentation with continual learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import datetime
import os
import numpy as np
from pathlib import Path

from vits.swin_unetr_model import create_swin_unetr_model
from continual_datasets.medical_segmentation import (
    build_continual_medical_dataloader,
    create_task_split_for_organs,
)
from engines.swin_unetr_engine import (
    train_one_epoch,
    evaluate,
    evaluate_till_now,
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

    # Create task splits for continual learning
    if hasattr(args, 'organ_mapping'):
        total_organs = len([k for k in args.organ_mapping.keys() if k > 0])  # Exclude background
    else:
        total_organs = args.out_channels - 1  # Exclude background

    task_organ_lists = create_task_split_for_organs(
        num_tasks=args.num_tasks,
        total_organs=total_organs
    )

    print(f"Continual learning task splits:")
    for i, organs in enumerate(task_organ_lists):
        print(f"  Task {i}: Organs {organs}")

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
            batch_size=1,  # Use batch size 1 for validation
            num_workers=args.num_workers,
            task_id=task_id,
            organ_list=task_organ_lists[task_id],
            mode='val',
            roi_size=args.roi_size,
            cache_rate=0.0,  # No cache for validation
        )

        data_loaders.append({
            'train': train_loader,
            'val': val_loader,
        })

    print(f"Data loaders created for {args.num_tasks} tasks")

    # Create model
    print(f"Creating Swin UNETR model")
    model = create_swin_unetr_model(args)
    model.to(device)

    # Optionally freeze backbone
    if args.freeze and 'encoder' in args.freeze:
        print("Freezing Swin UNETR encoder")
        model.freeze_backbone()

    # Distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters:,}')

    # Evaluation mode
    if args.eval:
        evaluate_all_tasks(
            model=model_without_ddp,
            data_loaders=data_loaders,
            device=device,
            organ_lists=task_organ_lists,
            args=args,
        )
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

        train_loader = data_loaders[task_id]['train']
        val_loader = data_loaders[task_id]['val']

        # Create optimizer for current task
        if args.reinit_optimizer:
            if args.larger_prompt_lr:
                # Separate learning rates for prompts and other parameters
                prompt_params = model_without_ddp.get_prompt_params()
                other_params = [p for p in model_without_ddp.parameters()
                               if p.requires_grad and p not in prompt_params]

                param_groups = [
                    {'params': prompt_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                    {'params': other_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay},
                ]
                optimizer = torch.optim.AdamW(param_groups)
            else:
                optimizer = torch.optim.AdamW(
                    [p for p in model_without_ddp.parameters() if p.requires_grad],
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    betas=args.opt_betas,
                    eps=args.opt_eps,
                )
        else:
            if task_id == 0:
                optimizer = torch.optim.AdamW(
                    [p for p in model_without_ddp.parameters() if p.requires_grad],
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
                step_size=args.decay_epochs,
                gamma=args.decay_rate,
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
                    model=model_without_ddp,
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
                        model=model_without_ddp,
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
                    model=model_without_ddp,
                    task_id=task_id,
                    epoch=epoch,
                    args=args,
                )

        # Save final checkpoint for this task
        save_checkpoint(
            model=model_without_ddp,
            task_id=task_id,
            epoch=args.epochs - 1,
            args=args,
            filename=f'task{task_id}_checkpoint.pth'
        )

        print(f"\nTask {task_id} training completed. Best Dice: {best_dice:.4f}")

        # Evaluate on all tasks seen so far
        print(f"\nEvaluating on all tasks up to Task {task_id}...")
        stat_matrix, avg_dice = evaluate_till_now(
            model=model_without_ddp,
            data_loaders=data_loaders,
            device=device,
            task_id=task_id,
            class_mask=None,
            organ_lists=task_organ_lists,
            args=args,
        )

        print(f"Average Dice Score across tasks 0-{task_id}: {avg_dice:.4f}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTotal training time: {total_time_str}")


def evaluate_all_tasks(model, data_loaders, device, organ_lists, args):
    """
    Evaluate model on all tasks

    Args:
        model: Model to evaluate
        data_loaders: List of data loaders
        device: Device
        organ_lists: List of organ lists for each task
        args: Arguments
    """
    print("\n" + "=" * 80)
    print("Evaluation Mode")
    print("=" * 80 + "\n")

    for task_id in range(args.num_tasks):
        checkpoint_path = os.path.join(
            args.output_dir,
            'checkpoint',
            f'task{task_id}_checkpoint.pth'
        )

        if os.path.exists(checkpoint_path):
            print(f'Loading checkpoint from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        else:
            print(f'No checkpoint found at: {checkpoint_path}')
            continue

        # Evaluate on this task
        val_loader = data_loaders[task_id]['val']
        results = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            task_id=task_id,
            organ_list=organ_lists[task_id],
            args=args,
        )

        print(f"Task {task_id} - Dice: {results['dice']:.4f}, "
              f"Hausdorff: {results['hausdorff']:.4f}\n")


def save_checkpoint(model, task_id, epoch, args, filename=None):
    """
    Save model checkpoint

    Args:
        model: Model to save
        task_id: Current task ID
        epoch: Current epoch
        args: Arguments
        filename: Optional filename (default: 'checkpoint.pth')
    """
    checkpoint_dir = Path(args.output_dir) / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f'task{task_id}_epoch{epoch}_checkpoint.pth'

    checkpoint_path = checkpoint_dir / filename

    checkpoint = {
        'model': model.state_dict(),
        'task_id': task_id,
        'epoch': epoch,
        'args': args,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    print("Swin UNETR trainer module")
    print("Use main.py to start training")
