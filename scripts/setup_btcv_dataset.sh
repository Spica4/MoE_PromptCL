#!/bin/bash
# BTCV Dataset Setup Guide

echo "=========================================="
echo "BTCV Multi-Organ Segmentation Dataset"
echo "=========================================="
echo
echo "⚠️  This dataset requires manual download and registration"
echo
echo "Steps to set up BTCV dataset:"
echo
echo "1. Register and download from Synapse:"
echo "   https://www.synapse.org/#!Synapse:syn3193805"
echo
echo "2. Download 'RawData.zip' (~24GB)"
echo
echo "3. Create directory structure:"
echo "   mkdir -p local_datasets/medical/BTCV"
echo
echo "4. Extract the dataset:"
echo "   unzip RawData.zip -d local_datasets/medical/BTCV/"
echo
echo "5. Organize the files:"
echo "   cd local_datasets/medical/BTCV"
echo "   mv RawData/Training/img imagesTr"
echo "   mv RawData/Training/label labelsTr"
echo
echo "Expected structure:"
echo "local_datasets/medical/BTCV/"
echo "├── imagesTr/"
echo "│   ├── img0001.nii.gz"
echo "│   ├── img0002.nii.gz"
echo "│   └── ... (30 files)"
echo "└── labelsTr/"
echo "    ├── label0001.nii.gz"
echo "    ├── label0002.nii.gz"
echo "    └── ... (30 files)"
echo
echo "=========================================="
echo
echo "For testing without real data, you can create dummy data:"
echo "  bash scripts/create_dummy_medical_data.sh"
echo
