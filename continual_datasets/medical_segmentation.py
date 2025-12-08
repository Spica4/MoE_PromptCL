"""
3D Medical Image Segmentation Dataset for Continual Learning
Supports BTCV and other NIfTI format datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from pathlib import Path
import sys
sys.path.append('..')
from utils.amos_mapping import generate_fixed_label_for_continual_learning, print_label_mapping


class MedicalSegmentationDataset(Dataset):
    """
    Dataset for 3D medical image segmentation with continual learning support
    """

    def __init__(
        self,
        data_dir,
        mode='train',
        task_id=None,
        organ_list=None,
        roi_size=(96, 96, 96),
        cache_rate=0.0,
        num_samples=None,
        transform=None
    ):
        """
        Args:
            data_dir: Root directory of the dataset
            mode: 'train', 'val', or 'test'
            task_id: Current task ID for continual learning
            organ_list: List of organ labels for current task
            roi_size: Size of the ROI for random crop
            cache_rate: Percentage of data to cache
            num_samples: Number of samples (None for all)
            transform: Additional transforms
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.task_id = task_id
        self.organ_list = organ_list
        self.roi_size = tuple(roi_size) if isinstance(roi_size, list) else roi_size
        self.cache_rate = cache_rate

        # Load data list
        self.data_list = self._load_data_list()

        if num_samples is not None:
            self.data_list = self.data_list[:num_samples]

        # Define transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform

    def _load_data_list(self):
        """Load list of image and label file paths"""
        data_list = []

        # BTCV dataset structure
        if self.mode == 'train':
            img_dir = self.data_dir / 'imagesTr'
            label_dir = self.data_dir / 'labelsTr'
        elif self.mode == 'val':
            img_dir = self.data_dir / 'imagesVa'
            label_dir = self.data_dir / 'labelsVa'
        else:  # test
            img_dir = self.data_dir / 'imagesTs'
            label_dir = self.data_dir / 'labelsTs'

        if not img_dir.exists():
            raise ValueError(f"Image directory {img_dir} does not exist")

        for img_path in sorted(img_dir.glob('*.nii.gz')):
            label_name = img_path.name.replace('img', 'label')
            label_path = label_dir / label_name

            # Skip empty files
            if img_path.stat().st_size == 0:
                print(f"Warning: Skipping empty image file: {img_path}")
                continue

            if label_path.exists():
                if label_path.stat().st_size == 0:
                    print(f"Warning: Skipping empty label file: {label_path}")
                    continue
                data_dict = {
                    'image': str(img_path),
                    'label': str(label_path),
                }
                data_list.append(data_dict)
            elif self.mode == 'test':
                data_dict = {
                    'image': str(img_path),
                    'label': None,
                }
                data_list.append(data_dict)

        return data_list

    def _get_default_transforms(self):
        """Get default MONAI transforms for 3D medical images"""

        # ラベルマッピングの準備（organ_listが指定されている場合）
        transform_list = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        ]

        # 継続学習用のラベルマッピングを追加
        if self.organ_list is not None:
            orig_labels, target_labels = generate_fixed_label_for_continual_learning(
                task_organ_ids=self.organ_list,
                total_organs=15,  # AMOS22は15臓器
                is_multi_label=True
            )
            if self.task_id == 0:  # 最初のタスクのみ表示
                print(f"\nTask {self.task_id} Label Mapping:")
                print_label_mapping(orig_labels, target_labels)

            transform_list.append(
                transforms.MapLabelValued(
                    keys=["label"],
                    orig_labels=orig_labels,
                    target_labels=target_labels,
                )
            )

        if self.mode == 'train':
            # Training transforms with augmentation
            transform_list.extend([
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.roi_size,
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=self.roi_size),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                transforms.ToTensord(keys=["image", "label"]),
            ])
        else:
            # Validation/test transforms (no augmentation)
            transform_list.append(transforms.ToTensord(keys=["image", "label"]))

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Apply transforms (including MapLabelValued)
        if self.transform is not None:
            data = self.transform(data)
            # Handle case where transform returns a list
            if isinstance(data, list):
                data = data[0]

        # Add task_id to data
        if self.task_id is not None:
            data['task_id'] = self.task_id

        return data


def build_continual_medical_dataloader(
    dataset_name,
    data_dir,
    batch_size,
    num_workers,
    task_id=None,
    organ_list=None,
    mode='train',
    roi_size=(96, 96, 96),
    cache_rate=0.0,
    **kwargs
):
    """
    Build dataloader for continual learning on 3D medical images

    Args:
        dataset_name: Name of dataset ('BTCV', etc.)
        data_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        task_id: Current task ID
        organ_list: List of organs for current task
        mode: 'train', 'val', or 'test'
        roi_size: ROI size for cropping
        cache_rate: Cache rate for faster loading

    Returns:
        DataLoader
    """
    # Create dataset
    dataset = MedicalSegmentationDataset(
        data_dir=data_dir,
        mode=mode,
        task_id=task_id,
        organ_list=organ_list,
        roi_size=roi_size,
        cache_rate=cache_rate,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )

    return dataloader


def create_task_split_for_organs(num_tasks=4, total_organs=15):
    """
    Create task splits for continual learning on multi-organ segmentation
    Each task is assigned one organ

    Args:
        num_tasks: Number of tasks
        total_organs: Total number of organ classes (excluding background)
                     Default is 15 for AMOS22

    Returns:
        List of organ lists for each task
    """
    task_splits = []
    for i in range(num_tasks):
        organ_id = i + 1
        if organ_id <= total_organs:
            task_splits.append([organ_id])
        else:
            task_splits.append([])

    return task_splits
