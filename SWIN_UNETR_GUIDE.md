# Swin UNETR for 3D Medical Image Segmentation

このガイドでは、Swin UNETRを使った3D医療画像セグメンテーションと継続学習の実装について説明します。

## 📋 目次

1. [学習済みモデル](#学習済みモデル)
2. [セットアップ](#セットアップ)
3. [使用方法](#使用方法)
4. [カスタマイズ](#カスタマイズ)

---

## 🎯 学習済みモデル

### MONAI公式の学習済みSwin UNETRモデル

| モデル | データセット | サイズ | 用途 | ダウンロード |
|--------|-------------|--------|------|-------------|
| **Self-Supervised** | 5,050 CT画像 | 約430MB | 継続学習の初期化に推奨 | [リンク](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) |
| **BTCV Fine-tuned** | BTCV 13臓器 | 約430MB | 参考用（ファインチューニング済み） | [リンク](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.btcv_f48_r.pt) |

### モデルの特徴

#### 1. **Self-Supervised Model (推奨)**
- **事前学習**: 5,050枚のCT画像で自己教師あり学習
- **メリット**:
  - 継続学習に適した汎用的な特徴表現
  - タスク固有のバイアスがない
  - NoRGaプロンプトと組み合わせて効果的
- **使用ケース**: 新しいデータセットでの継続学習

#### 2. **BTCV Fine-tuned Model**
- **事前学習**: BTCV Multi-Organ Segmentationでファインチューニング
- **メリット**:
  - BTCV特有の高精度
  - すぐに使える性能
- **使用ケース**: BTCVデータセットでのベンチマーク、転移学習

---

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

必要なパッケージ：
- `monai==1.3.0` - Swin UNETRモデル
- `nibabel==5.2.0` - NIfTI医療画像形式
- `SimpleITK==2.3.1` - 医療画像処理

### 2. 学習済みモデルのダウンロード

**自動ダウンロード（推奨）:**

```bash
bash scripts/download_swin_unetr_weights.sh
```

**手動ダウンロード:**

```bash
# ディレクトリ作成
mkdir -p checkpoints/swin_unetr

# Self-supervised model
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt \
    -O checkpoints/swin_unetr/model_swinvit.pt

# BTCV fine-tuned model
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.btcv_f48_r.pt \
    -O checkpoints/swin_unetr/swin_unetr_btcv.pt
```

### 3. データセットの準備

**BTCV Multi-Organ Segmentation Dataset:**

1. データセットをダウンロード: https://www.synapse.org/#!Synapse:syn3193805
2. 以下の構造で配置:

```
./local_datasets/medical/BTCV/
├── imagesTr/
│   ├── img0001.nii.gz
│   ├── img0002.nii.gz
│   └── ...
└── labelsTr/
    ├── label0001.nii.gz
    ├── label0002.nii.gz
    └── ...
```

**他のデータセット:**
- BraTS: 脳腫瘍セグメンテーション
- Medical Segmentation Decathlon: 10種類の医療画像タスク

---

## 💻 使用方法

### 1. クイックスタート

```bash
# 学習済みモデルをダウンロード
bash scripts/download_swin_unetr_weights.sh

# クイックテスト（3タスク、5エポック）
bash scripts/medical_BTCV_SwinUNETR_quick.sh
```

### 2. 本格的なトレーニング

```bash
# 5タスク、100エポック
bash scripts/medical_BTCV_SwinUNETR.sh
```

### 3. カスタムトレーニング

```bash
python main.py swin_unetr_norgaprompt \
    --dataset BTCV \
    --data-path ./local_datasets/medical/BTCV \
    --output_dir ./output/my_experiment \
    --batch-size 2 \
    --epochs 100 \
    --num_tasks 5 \
    --pretrained True \
    --pretrained_path ./checkpoints/swin_unetr/model_swinvit.pt \
    --seed 42
```

### 4. 評価のみ実行

```bash
python main.py swin_unetr_norgaprompt \
    --dataset BTCV \
    --data-path ./local_datasets/medical/BTCV \
    --output_dir ./output/swin_unetr_btcv \
    --eval
```

---

## ⚙️ カスタマイズ

### 1. 異なるデータセットを使用

#### BraTSデータセット（脳腫瘍）

```bash
python main.py swin_unetr_norgaprompt \
    --dataset BraTS \
    --data-path ./local_datasets/medical/BraTS \
    --out_channels 4 \  # 4クラス（背景+3腫瘍領域）
    --num_tasks 3 \
    ...
```

### 2. 画像サイズとバッチサイズの調整

メモリに応じて調整：

```bash
# より大きい画像サイズ（より高精度、より多くのメモリ）
--img_size 128 128 128 \
--batch-size 1 \

# より小さい画像サイズ（高速、メモリ節約）
--img_size 64 64 64 \
--batch-size 4 \
```

### 3. NoRGaプロンプトパラメータ

```bash
# プロンプト長を調整
--length 20 \  # より長いプロンプト

# ゲート活性化関数を変更
--gate_act sigmoid \  # tanh or sigmoid

# プロンプトプール設定
--size 10 \  # タスク数に合わせて調整
--top_k 2 \  # 複数プロンプト選択
```

### 4. 損失関数の選択

```bash
# Dice Loss のみ
--loss_type dice

# Dice + Cross-Entropy（デフォルト、推奨）
--loss_type dice_ce \
--dice_loss_weight 0.5 \
--ce_loss_weight 0.5

# Dice + Focal Loss
--loss_type dice_focal
```

### 5. 学習率スケジューリング

```bash
# Cosineスケジューラ（デフォルト）
--sched cosine \
--lr 1e-4 \
--min-lr 1e-6 \
--warmup-epochs 10

# Stepスケジューラ
--sched step \
--decay-epochs 30 \
--decay-rate 0.1
```

---

## 📊 期待される性能

### BTCVデータセット

| タスク | 臓器 | 期待Dice Score |
|--------|------|---------------|
| Task 0 | Spleen, Right Kidney, Left Kidney | ~0.85 |
| Task 1 | Gallbladder, Esophagus, Liver | ~0.82 |
| Task 2 | Stomach, Aorta | ~0.80 |
| Task 3 | Postcava, Portal Vein | ~0.78 |
| Task 4 | Pancreas, Adrenals | ~0.75 |
| **Average** | - | **~0.80** |

*注: 実際の性能はデータセット、ハイパーパラメータ、学習時間によって変動します*

---

## 🔧 トラブルシューティング

### Out of Memory (OOM) エラー

```bash
# 解決策1: バッチサイズを減らす
--batch-size 1

# 解決策2: 画像サイズを小さくする
--img_size 64 64 64

# 解決策3: Gradient Checkpointingを有効化
--use_checkpoint True

# 解決策4: より少ないワーカー
--num_workers 2
```

### 学習済みモデルが読み込めない

```bash
# チェックポイントの存在確認
ls -lh checkpoints/swin_unetr/

# 再ダウンロード
bash scripts/download_swin_unetr_weights.sh

# スクラッチから学習
--pretrained False
```

### データセットが見つからない

```bash
# データパスを確認
ls -la ./local_datasets/medical/BTCV/

# 正しいパスを指定
--data-path /path/to/your/BTCV/dataset
```

---

## 📚 参考文献

1. **Swin UNETR Paper**:
   - Tang et al. "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis" (CVPR 2022)
   - https://arxiv.org/abs/2201.01266

2. **MONAI Framework**:
   - https://docs.monai.io/
   - https://github.com/Project-MONAI/MONAI

3. **NoRGa (本リポジトリの手法)**:
   - Le et al. "Mixture of Experts Meets Prompt-Based Continual Learning" (NeurIPS 2024)
   - https://arxiv.org/abs/2405.14124

---

## 💡 ベストプラクティス

1. **学習済みモデルの使用**: Self-supervised modelから始めることを推奨
2. **データ拡張**: デフォルトの拡張設定は良好（必要に応じて調整）
3. **検証頻度**: `--val_interval 5` で定期的に検証
4. **チェックポイント**: 重要な実験は複数シードで実行
5. **メモリ管理**: 大きなボリュームには sliding window inference を使用（自動）

---

## 🤝 サポート

問題が発生した場合:
1. このガイドのトラブルシューティングセクションを確認
2. GitHubのIssuesで質問
3. MONAI公式ドキュメントを参照

---

**Happy Training! 🚀**
