"""
Batch inference script for multiple 3D medical images
"""

import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.data import CacheDataset, DataLoader
from vits.swin_unetr_model import create_swin_unetr_model


def get_inference_transforms():
    """Get transforms for inference"""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.EnsureTyped(keys=["image"]),
    ])


def main():
    parser = argparse.ArgumentParser('Batch Medical Image Segmentation Inference')

    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input .nii.gz images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predicted .nii.gz segmentations')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Model parameters
    parser.add_argument('--img_size', default=[96, 96, 96], type=int, nargs=3,
                        help='Input image size (D H W)')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels')
    parser.add_argument('--out_channels', default=14, type=int,
                        help='Number of output classes')
    parser.add_argument('--feature_size', default=48, type=int,
                        help='Feature size for Swin transformer')

    # Inference parameters
    parser.add_argument('--roi_size', default=[96, 96, 96], type=int, nargs=3,
                        help='ROI size for sliding window inference')
    parser.add_argument('--sw_batch_size', default=4, type=int,
                        help='Batch size for sliding window')
    parser.add_argument('--overlap', default=0.5, type=float,
                        help='Overlap ratio for sliding window')

    # Device
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of input images
    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.glob('*.nii.gz'))

    if len(image_files) == 0:
        raise ValueError(f"No .nii.gz files found in {input_dir}")

    print(f"Found {len(image_files)} images to process")

    # Create model
    print("Creating model...")
    model = create_swin_unetr_model(
        img_size=tuple(args.img_size),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
    )
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully")

    # Create data list
    data_list = [{"image": str(img_path)} for img_path in image_files]

    # Create dataset and dataloader
    transforms_fn = get_inference_transforms()
    dataset = CacheDataset(
        data=data_list,
        transform=transforms_fn,
        cache_rate=0.0,
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    # Run inference
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(dataloader, desc="Processing images")):
            image = batch_data["image"].to(device)
            image_path = image_files[idx]

            # Sliding window inference
            predictions = sliding_window_inference(
                inputs=image,
                roi_size=tuple(args.roi_size),
                sw_batch_size=args.sw_batch_size,
                predictor=lambda x: model(x, task_id=-1, train=False)[0],
                overlap=args.overlap,
                mode="gaussian",
            )

            # Get predicted labels
            pred_labels = torch.argmax(predictions, dim=1).squeeze(0)
            pred_labels = pred_labels.cpu().numpy().astype(np.uint8)

            # Load original image for affine
            original_img = nib.load(str(image_path))

            # Create and save prediction
            pred_nifti = nib.Nifti1Image(pred_labels, affine=original_img.affine)
            output_path = output_dir / f"{image_path.stem}_pred.nii.gz"
            nib.save(pred_nifti, str(output_path))

            print(f"[{idx+1}/{len(image_files)}] Saved: {output_path.name} | Unique labels: {np.unique(pred_labels)}")

    print(f"\nBatch inference completed! All predictions saved to: {output_dir}")


if __name__ == '__main__':
    main()
