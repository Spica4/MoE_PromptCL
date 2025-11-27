"""
Medical image segmentation dataset for continual learning
Supports BTCV, BraTS, and other 3D medical imaging datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms
from monai.data import CacheDataset, DataLoader, Dataset as MonaiDataset
import nibabel as nib
from pathlib import Path


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
            task_id: Current task ID for continual learning (None for all tasks)
            organ_list: List of organ labels for current task
            roi_size: Size of the ROI for random crop
            cache_rate: Percentage of data to cache (0.0-1.0)
            num_samples: Number of samples (None for all)
            transform: Additional transforms
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.task_id = task_id
        self.organ_list = organ_list
        # Convert to tuple if necessary (for MONAI transforms)
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
        """
        Load list of image and label file paths
        Override this method for specific datasets
        """
        data_list = []

        # Example structure for BTCV dataset
        img_dir = self.data_dir / 'imagesTr' if self.mode == 'train' else self.data_dir / 'imagesTs'
        label_dir = self.data_dir / 'labelsTr' if self.mode == 'train' else self.data_dir / 'labelsTs'

        if not img_dir.exists():
            raise ValueError(f"Image directory {img_dir} does not exist")

        for img_path in sorted(img_dir.glob('*.nii.gz')):
            # Handle different naming conventions (e.g., img0001.nii.gz -> label0001.nii.gz)
            label_name = img_path.name.replace('img', 'label')
            label_path = label_dir / label_name

            # Skip if files are empty or corrupted
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
        """
        Get default MONAI transforms for medical images
        """
        if self.mode == 'train':
            # Training transforms with augmentation
            transform = transforms.Compose([
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
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
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
            transform = transforms.Compose([
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
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ])

        return transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)
            # Handle case where transform returns a list (e.g., RandCropByPosNegLabeld)
            if isinstance(data, list):
                data = data[0]

        # Filter labels for current task if specified
        if self.task_id is not None and self.organ_list is not None:
            label = data['label']
            mask = torch.zeros_like(label)
            for organ_id in self.organ_list:
                mask[label == organ_id] = organ_id
            data['label'] = mask

        # Add task_id to data
        if self.task_id is not None:
            data['task_id'] = self.task_id

        return data


class BTCVDataset(MedicalSegmentationDataset):
    """
    BTCV Multi-Organ Segmentation Dataset
    https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

    13 organs: Spleen, Right Kidney, Left Kidney, Gallbladder, Esophagus,
               Liver, Stomach, Aorta, Postcava, Portal Vein, Pancreas,
               Right Adrenal, Left Adrenal
    """

    ORGAN_MAPPING = {
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
        13: 'left_adrenal',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 14  # including background


class BraTSDataset(MedicalSegmentationDataset):
    """
    BraTS Brain Tumor Segmentation Dataset
    https://www.med.upenn.edu/cbica/brats2020/

    4 regions: Necrotic tumor core, Peritumoral edema,
               GD-enhancing tumor, Whole tumor
    """

    TUMOR_MAPPING = {
        0: 'background',
        1: 'necrotic_core',
        2: 'edema',
        3: 'enhancing_tumor',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 4  # including background

    def _load_data_list(self):
        """
        BraTS has 4 modalities: T1, T1ce, T2, FLAIR
        """
        data_list = []

        subject_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        for subject_dir in subject_dirs:
            t1_path = subject_dir / f"{subject_dir.name}_t1.nii.gz"
            t1ce_path = subject_dir / f"{subject_dir.name}_t1ce.nii.gz"
            t2_path = subject_dir / f"{subject_dir.name}_t2.nii.gz"
            flair_path = subject_dir / f"{subject_dir.name}_flair.nii.gz"
            seg_path = subject_dir / f"{subject_dir.name}_seg.nii.gz"

            if all([p.exists() for p in [t1_path, t1ce_path, t2_path, flair_path]]):
                data_dict = {
                    'image': [str(t1_path), str(t1ce_path), str(t2_path), str(flair_path)],
                    'label': str(seg_path) if seg_path.exists() else None,
                }
                data_list.append(data_dict)

        return data_list


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
    Build dataloader for continual learning on medical images

    Args:
        dataset_name: Name of dataset ('BTCV', 'BraTS', etc.)
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
    # Select dataset class
    if dataset_name.upper() == 'BTCV':
        dataset_class = BTCVDataset
    elif dataset_name.upper() == 'BRATS':
        dataset_class = BraTSDataset
    else:
        dataset_class = MedicalSegmentationDataset

    # Create dataset
    dataset = dataset_class(
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


def create_task_split_for_organs(num_tasks=5, total_organs=13):
    """
    Create task splits for continual learning on multi-organ segmentation

    Args:
        num_tasks: Number of tasks to split into
        total_organs: Total number of organ classes (excluding background)

    Returns:
        List of organ lists for each task
    """
    organs = list(range(1, total_organs + 1))  # 1 to total_organs
    organs_per_task = len(organs) // num_tasks

    task_splits = []
    for i in range(num_tasks):
        start_idx = i * organs_per_task
        if i == num_tasks - 1:
            # Last task gets remaining organs
            end_idx = len(organs)
        else:
            end_idx = (i + 1) * organs_per_task

        task_splits.append(organs[start_idx:end_idx])

    return task_splits


# Example usage
if __name__ == "__main__":
    # Create task splits for 5 tasks
    task_splits = create_task_split_for_organs(num_tasks=5, total_organs=13)
    print("Task splits for continual learning:")
    for i, organs in enumerate(task_splits):
        print(f"Task {i}: Organs {organs}")

    # Create dataloader for task 0
    # dataloader = build_continual_medical_dataloader(
    #     dataset_name='BTCV',
    #     data_dir='/local_datasets/medical/BTCV',
    #     batch_size=2,
    #     num_workers=4,
    #     task_id=0,
    #     organ_list=task_splits[0],
    #     mode='train',
    # )
