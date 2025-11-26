#!/bin/bash
# Create dummy medical imaging data for testing

set -e

echo "=========================================="
echo "Creating Dummy BTCV Dataset for Testing"
echo "=========================================="
echo

# Create directory structure
BTCV_DIR="./local_datasets/medical/BTCV"
mkdir -p ${BTCV_DIR}/imagesTr
mkdir -p ${BTCV_DIR}/labelsTr

echo "Creating dummy NIfTI files..."
echo "This requires Python with numpy and nibabel installed."
echo

# Create dummy data using Python
python3 << 'EOF'
import os
import numpy as np
try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel not installed.")
    print("Please install: pip install nibabel")
    exit(1)

# Create dummy 3D volumes
print("Generating 5 dummy CT volumes...")

for i in range(1, 6):  # Create 5 samples for testing
    # Create random 3D image (small size for quick testing)
    img_data = np.random.randint(-200, 200, (64, 64, 64), dtype=np.int16)

    # Create random label (14 classes including background)
    label_data = np.random.randint(0, 14, (64, 64, 64), dtype=np.uint8)

    # Create NIfTI images
    img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))
    label_nii = nib.Nifti1Image(label_data, affine=np.eye(4))

    # Save files
    img_path = f"./local_datasets/medical/BTCV/imagesTr/img{i:04d}.nii.gz"
    label_path = f"./local_datasets/medical/BTCV/labelsTr/label{i:04d}.nii.gz"

    nib.save(img_nii, img_path)
    nib.save(label_nii, label_path)

    print(f"  Created: img{i:04d}.nii.gz and label{i:04d}.nii.gz")

print("\nDummy dataset created successfully!")
print(f"Location: ./local_datasets/medical/BTCV/")
EOF

echo
echo "=========================================="
echo "Dummy Dataset Created!"
echo "=========================================="
echo
echo "Structure:"
tree -L 3 ./local_datasets/medical/BTCV/ 2>/dev/null || ls -R ./local_datasets/medical/BTCV/

echo
echo "⚠️  Note: This is DUMMY data for testing only!"
echo "   - Random noise images (64x64x64)"
echo "   - Random segmentation labels"
echo "   - Not suitable for real training"
echo
echo "For real experiments, download actual BTCV data:"
echo "  bash scripts/setup_btcv_dataset.sh"
echo
echo "You can now test the code with:"
echo "  bash scripts/medical_BTCV_SwinUNETR_quick.sh"
echo
