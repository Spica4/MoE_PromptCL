#!/bin/bash
# データセットダウンロードスクリプト

set -e

echo "=========================================="
echo "データセットダウンロード"
echo "=========================================="
echo

# ディレクトリ作成
mkdir -p local_datasets
cd local_datasets

echo "📦 ダウンロード可能なデータセット:"
echo
echo "1. CIFAR-100"
echo "   wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
echo "   tar -xzf cifar-100-python.tar.gz"
echo
echo "2. ImageNet-R"
echo "   wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
echo "   tar -xf imagenet-r.tar"
echo
echo "3. CUB-200"
echo "   wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
echo "   tar -xzf CUB_200_2011.tgz"
echo
echo "4. 5-Datasets (SVHN, MNIST, CIFAR10, NotMNIST, FashionMNIST)"
echo "   これらはPyTorchが自動ダウンロードします"
echo
echo "=========================================="
echo "医療画像データセット (Swin UNETR用):"
echo "=========================================="
echo
echo "1. BTCV Multi-Organ Segmentation"
echo "   手動ダウンロード必要（要登録）:"
echo "   https://www.synapse.org/#!Synapse:syn3193805"
echo
echo "2. BraTS Brain Tumor Segmentation"
echo "   手動ダウンロード必要（要登録）:"
echo "   https://www.med.upenn.edu/cbica/brats2020/"
echo
echo "=========================================="

# ユーザー選択
read -p "ダウンロードを開始しますか？ [1:CIFAR-100, 2:ImageNet-R, 3:CUB-200, 0:スキップ] " choice

case $choice in
    1)
        echo "CIFAR-100 をダウンロード中..."
        wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
        tar -xzf cifar-100-python.tar.gz
        rm cifar-100-python.tar.gz
        echo "✓ CIFAR-100 ダウンロード完了"
        ;;
    2)
        echo "ImageNet-R をダウンロード中..."
        wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
        tar -xf imagenet-r.tar
        rm imagenet-r.tar
        echo "✓ ImageNet-R ダウンロード完了"
        ;;
    3)
        echo "CUB-200 をダウンロード中..."
        wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
        tar -xzf CUB_200_2011.tgz
        rm CUB_200_2011.tgz
        echo "✓ CUB-200 ダウンロード完了"
        ;;
    0)
        echo "スキップしました"
        ;;
    *)
        echo "無効な選択"
        ;;
esac

cd ..

echo
echo "=========================================="
echo "現在のデータセット構造:"
echo "=========================================="
tree -L 2 local_datasets 2>/dev/null || ls -la local_datasets

echo
echo "完了！"
