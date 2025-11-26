# データセット配置ガイド

このドキュメントでは、MoE_PromptCLリポジトリで使用するデータセットの配置方法を説明します。

## 📂 ディレクトリ構造概要

```
MoE_PromptCL/
├── local_datasets/              # すべてのデータセットのルート
│   ├── cifar-100-python/        # CIFAR-100
│   ├── imagenet-r/              # ImageNet-R
│   ├── CUB_200_2011/            # CUB-200
│   ├── SVHN/                    # 5-datasets用
│   ├── MNIST/                   # 5-datasets用
│   └── medical/                 # 医療画像データセット
│       ├── BTCV/                # BTCV Multi-organ
│       └── BraTS/               # Brain Tumor
└── checkpoints/                 # 学習済みモデル
    ├── ibot_vitbase16_pretrain.pth
    ├── dino_vitbase16_pretrain.pth
    └── swin_unetr/
        ├── model_swinvit.pt
        └── swin_unetr_btcv.pt
```

---

## 📦 1. 元のコード用データセット（2D画像分類）

### CIFAR-100

**ダウンロードとセットアップ:**

```bash
cd MoE_PromptCL
mkdir -p local_datasets
cd local_datasets

# ダウンロード
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

# 解凍
tar -xzf cifar-100-python.tar.gz

# クリーンアップ
rm cifar-100-python.tar.gz
```

**期待されるディレクトリ構造:**
```
local_datasets/
└── cifar-100-python/
    ├── meta
    ├── train
    └── test
```

**使用例:**
```bash
bash scripts/cifar100_Sup21k_NoRGa.sh
```

---

### ImageNet-R

**ダウンロードとセットアップ:**

```bash
cd local_datasets

# ダウンロード (約4GB)
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar

# 解凍
tar -xf imagenet-r.tar

# クリーンアップ
rm imagenet-r.tar
```

**期待されるディレクトリ構造:**
```
local_datasets/
└── imagenet-r/
    ├── n01443537/    # クラスフォルダ
    ├── n01484850/
    └── ...
```

**使用例:**
```bash
bash scripts/imr_Sup21k_NoRGa.sh
```

---

### CUB-200

**ダウンロードとセットアップ:**

```bash
cd local_datasets

# ダウンロード (約1.1GB)
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

# 解凍
tar -xzf CUB_200_2011.tgz

# クリーンアップ
rm CUB_200_2011.tgz
```

**期待されるディレクトリ構造:**
```
local_datasets/
└── CUB_200_2011/
    ├── images/
    │   ├── 001.Black_footed_Albatross/
    │   └── ...
    ├── images.txt
    ├── train_test_split.txt
    └── ...
```

**使用例:**
```bash
bash scripts/cub_Sup21k_NoRGa.sh
```

---

### 5-Datasets (SVHN, MNIST, CIFAR-10, NotMNIST, FashionMNIST)

**セットアップ:**

これらのデータセットは**PyTorchが自動的にダウンロード**します。
初回実行時に自動的に `local_datasets/` 配下にダウンロードされます。

```bash
# 初回実行時に自動ダウンロード
bash scripts/5datasets_Sup21k_NoRGa.sh
```

**期待されるディレクトリ構造（自動生成）:**
```
local_datasets/
├── SVHN/
├── MNIST/
├── CIFAR10/
├── NotMNIST/
└── FashionMNIST/
```

---

## 🏥 2. 医療画像データセット（3D Swin UNETR用）

### BTCV Multi-Organ Segmentation

**ダウンロード:**

⚠️ **登録が必要です**: https://www.synapse.org/#!Synapse:syn3193805

1. Synapseにアカウント登録
2. データ使用許諾に同意
3. ダウンロード: `RawData.zip` (約24GB)

**セットアップ:**

```bash
# 解凍先ディレクトリを作成
mkdir -p local_datasets/medical/BTCV

# ダウンロードしたファイルを解凍
unzip RawData.zip -d local_datasets/medical/BTCV/

# ファイル構造を整理（必要に応じて）
cd local_datasets/medical/BTCV
# RawData/Training/img → imagesTr
# RawData/Training/label → labelsTr
mv RawData/Training/img imagesTr
mv RawData/Training/label labelsTr
```

**期待されるディレクトリ構造:**

```
local_datasets/medical/BTCV/
├── imagesTr/
│   ├── img0001.nii.gz   # CT画像 (30例)
│   ├── img0002.nii.gz
│   └── ...
└── labelsTr/
    ├── label0001.nii.gz  # セグメンテーションラベル
    ├── label0002.nii.gz
    └── ...
```

**データセット詳細:**
- **画像数**: 30例（Training）
- **モダリティ**: CT
- **ラベル**: 13臓器
  1. Spleen (脾臓)
  2. Right Kidney (右腎臓)
  3. Left Kidney (左腎臓)
  4. Gallbladder (胆嚢)
  5. Esophagus (食道)
  6. Liver (肝臓)
  7. Stomach (胃)
  8. Aorta (大動脈)
  9. Postcava (下大静脈)
  10. Portal Vein (門脈)
  11. Pancreas (膵臓)
  12. Right Adrenal Gland (右副腎)
  13. Left Adrenal Gland (左副腎)

**使用例:**
```bash
bash scripts/medical_BTCV_SwinUNETR.sh
```

---

### BraTS Brain Tumor Segmentation

**ダウンロード:**

⚠️ **登録が必要です**: https://www.med.upenn.edu/cbica/brats2020/

**セットアップ:**

```bash
# 解凍先ディレクトリを作成
mkdir -p local_datasets/medical/BraTS

# ダウンロードしたファイルを解凍
tar -xzf MICCAI_BraTS2020_TrainingData.tar.gz -C local_datasets/medical/BraTS/
```

**期待されるディレクトリ構造:**

```
local_datasets/medical/BraTS/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
└── ...
```

**データセット詳細:**
- **画像数**: 369例（2020年版）
- **モダリティ**: MRI (4チャンネル: T1, T1ce, T2, FLAIR)
- **ラベル**: 4クラス
  0. Background (背景)
  1. Necrotic tumor core (壊死性腫瘍コア)
  2. Peritumoral edema (腫瘍周辺浮腫)
  3. GD-enhancing tumor (造影増強腫瘍)

**使用例:**
```bash
python main.py swin_unetr_norgaprompt \
    --dataset BraTS \
    --data-path ./local_datasets/medical/BraTS \
    --out_channels 4 \
    --num_tasks 3
```

---

## 🚀 クイックセットアップスクリプト

### 自動ダウンロードスクリプト（2Dデータセット用）

```bash
bash scripts/download_datasets.sh
```

このスクリプトは以下をダウンロードできます:
- CIFAR-100
- ImageNet-R
- CUB-200

### 学習済みモデルのダウンロード

```bash
# 2D ViTモデル (元のコード用)
mkdir -p checkpoints

# iBOT
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth \
    -O checkpoints/ibot_vitbase16_pretrain.pth

# DINO
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
    -O checkpoints/dino_vitbase16_pretrain.pth

# Swin UNETR (3D医療画像用)
bash scripts/download_swin_unetr_weights.sh
```

---

## ✅ セットアップ確認

すべてのデータセットが正しく配置されているか確認:

```bash
# ディレクトリ構造を確認
tree -L 3 local_datasets

# または
ls -R local_datasets
```

**期待される出力:**

```
local_datasets/
├── cifar-100-python/
│   ├── meta
│   ├── train
│   └── test
├── imagenet-r/
│   └── [200 class folders]
├── CUB_200_2011/
│   ├── images/
│   └── [metadata files]
└── medical/
    ├── BTCV/
    │   ├── imagesTr/
    │   └── labelsTr/
    └── BraTS/
        └── [patient folders]
```

---

## 📊 データセットサイズ

| データセット | サイズ | 画像数 | 備考 |
|------------|--------|--------|------|
| CIFAR-100 | ~170MB | 60,000 | 自動分割 |
| ImageNet-R | ~4GB | 30,000 | 200クラス |
| CUB-200 | ~1.1GB | 11,788 | 200鳥種 |
| BTCV | ~24GB | 30 | CT、13臓器 |
| BraTS | ~60GB | 369 | MRI、脳腫瘍 |

---

## 🔧 トラブルシューティング

### データセットが見つからない

```bash
# パスを確認
ls -la local_datasets/

# 設定ファイルでパスを指定
python main.py config_name --data-path /your/custom/path
```

### ダウンロードが失敗する

```bash
# wget の代わりに curl を使用
curl -L -O <URL>

# 再開可能なダウンロード
wget -c <URL>
```

### ディスク容量不足

```bash
# 必要なディスク容量を確認
df -h

# 不要なファイルを削除
rm local_datasets/*.tar.gz
rm local_datasets/*.zip
```

---

## 📚 参考リンク

### 2D画像分類データセット
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-R](https://github.com/hendrycks/imagenet-r)
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

### 3D医療画像データセット
- [BTCV (Synapse)](https://www.synapse.org/#!Synapse:syn3193805)
- [BraTS](https://www.med.upenn.edu/cbica/brats2020/)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

---

## 💡 ヒント

1. **データセットを分けて管理**: 2Dと3Dを別フォルダに
2. **シンボリックリンク**: 大きなデータセットは別ディスクに配置してリンク
   ```bash
   ln -s /mnt/data/BTCV local_datasets/medical/BTCV
   ```
3. **データセット検証**: トレーニング前にデータが正しく読み込めるか確認
   ```bash
   python -c "from continual_datasets.medical_segmentation import BTCVDataset; print('OK')"
   ```

---

**セットアップ完了後は、トレーニングを開始できます！** 🚀
