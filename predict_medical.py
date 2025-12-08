"""
Inference script for 3D medical image segmentation using Swin UNETR
"""

import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from monai.inferers import sliding_window_inference
from monai import transforms
from vits.swin_unetr_model import create_swin_unetr_model


def get_inference_transforms(img_size=(96, 96, 96)):
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


def predict_single_image(
    model,
    image_path,
    output_path,
    device,
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    overlap=0.5,
):
    """
    Predict segmentation for a single .nii.gz image

    Args:
        model: Trained model
        image_path: Path to input .nii.gz image
        output_path: Path to save predicted .nii.gz segmentation
        device: Device to run inference on
        roi_size: ROI size for sliding window inference
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio for sliding window
    """
    print(f"Loading image: {image_path}")

    # Load and preprocess image
    data_dict = {"image": str(image_path)}
    transforms_fn = get_inference_transforms()
    data = transforms_fn(data_dict)

    # Get image tensor
    image = data["image"].unsqueeze(0).to(device)  # Add batch dimension

    print(f"Input image shape: {image.shape}")

    # Run inference
    model.eval()
    with torch.no_grad():
        # Use sliding window inference for large volumes
        predictions = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: model(x, task_id=-1, train=False)[0],
            overlap=overlap,
            mode="gaussian",
        )

        # Get predicted labels (argmax over channels)
        pred_labels = torch.argmax(predictions, dim=1).squeeze(0)  # Remove batch dim
        pred_labels = pred_labels.cpu().numpy().astype(np.uint8)

    print(f"Prediction shape: {pred_labels.shape}")

    # Load original image to get affine matrix
    original_img = nib.load(str(image_path))

    # Create NIfTI image with prediction
    pred_nifti = nib.Nifti1Image(pred_labels, affine=original_img.affine)

    # Save prediction
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(pred_nifti, str(output_path))

    print(f"Prediction saved to: {output_path}")
    print(f"Unique labels in prediction: {np.unique(pred_labels)}")

    return pred_labels


def main():
    parser = argparse.ArgumentParser('Medical Image Segmentation Inference')

    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .nii.gz image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save predicted .nii.gz segmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Model parameters
    parser.add_argument('--img_size', default=[96, 96, 96], type=int, nargs=3,
                        help='Input image size (D H W)')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels')
    parser.add_argument('--out_channels', default=16, type=int,
                        help='Number of output classes (15 organs + background for AMOS22)')
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

    # Run prediction
    predict_single_image(
        model=model,
        image_path=args.input,
        output_path=args.output,
        device=device,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
    )

    print("Inference completed!")


if __name__ == '__main__':
    main()
