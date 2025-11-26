# クイックスタートガイド

このガイドでは、Swin UNETRをすぐに試す方法を説明します。

## ⚠️ 現在のエラーについて

```
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

このエラーは、**BTCVデータセットが配置されていない**ために発生しています。

---

## 🚀 解決方法

### **方法1: BTCVデータセットをダウンロード（本番用）**

#### **ステップ1: 登録とダウンロード**

1. Synapseに登録: https://www.synapse.org/
2. データセットページにアクセス: https://www.synapse.org/#!Synapse:syn3193805
3. データ使用許諾に同意
4. `RawData.zip` をダウンロード（約24GB）

#### **ステップ2: データセットを配置**

```bash
# ディレクトリ作成
mkdir -p local_datasets/medical/BTCV

# ダウンロードしたファイルを解凍
unzip ~/Downloads/RawData.zip -d local_datasets/medical/BTCV/

# ファイルを整理
cd local_datasets/medical/BTCV
mv RawData/Training/img imagesTr
mv RawData/Training/label labelsTr
cd ../../..
```

#### **ステップ3: 構造を確認**

```bash
ls -la local_datasets/medical/BTCV/
```

**期待される出力:**
```
local_datasets/medical/BTCV/
├── imagesTr/
│   ├── img0001.nii.gz
│   ├── img0002.nii.gz
│   └── ... (30 files)
└── labelsTr/
    ├── label0001.nii.gz
    ├── label0002.nii.gz
    └── ... (30 files)
```

---

### **方法2: ダミーデータで動作確認（テスト用）**

実際のデータがない場合は、小さなダミーデータで動作確認できます：

```bash
# Python環境でnibabelがインストールされていることを確認
pip install nibabel numpy

# ダミーデータ作成（5サンプル、64x64x64サイズ）
bash scripts/create_dummy_medical_data.sh
```

**注意:** ダミーデータはランダムノイズなので、実際の学習には使えません。

---

## ✅ セットアップ確認

データセットが正しく配置されているか確認：

```bash
# ファイル数を確認
echo "Images: $(ls local_datasets/medical/BTCV/imagesTr/*.nii.gz 2>/dev/null | wc -l)"
echo "Labels: $(ls local_datasets/medical/BTCV/labelsTr/*.nii.gz 2>/dev/null | wc -l)"
```

**期待される出力:**
```
Images: 30  (実データ) または 5 (ダミーデータ)
Labels: 30  (実データ) または 5 (ダミーデータ)
```

---

## 🎯 トレーニング開始

データセットの準備ができたら：

### **クイックテスト（推奨）**

```bash
bash scripts/medical_BTCV_SwinUNETR_quick.sh
```

設定:
- エポック数: 5
- タスク数: 3
- バッチサイズ: 1

### **本格トレーニング**

```bash
bash scripts/medical_BTCV_SwinUNETR.sh
```

設定:
- エポック数: 100
- タスク数: 5
- バッチサイズ: 2

---

## 🔧 よくあるトラブル

### **1. データセットが見つからない**

```bash
# パスを確認
ls -la local_datasets/medical/BTCV/

# 正しいパスを指定
python main.py swin_unetr_norgaprompt \
    --data-path /your/actual/path/to/BTCV \
    ...
```

### **2. メモリ不足**

```bash
# バッチサイズを減らす
--batch-size 1

# 画像サイズを小さくする
--img_size 64 64 64

# ワーカー数を減らす
--num_workers 1
```

### **3. nibabelがインストールされていない**

```bash
pip install nibabel SimpleITK
```

---

## 📚 関連ドキュメント

- **DATASET_SETUP.md** - データセット配置の詳細
- **SWIN_UNETR_GUIDE.md** - Swin UNETRの完全ガイド
- **README.md** - プロジェクト概要

---

## 💡 次のステップ

1. ✅ データセットを配置
2. ✅ 依存パッケージをインストール (`pip install -r requirements.txt`)
3. ✅ 学習済みモデルをダウンロード (`bash scripts/download_swin_unetr_weights.sh`)
4. ✅ クイックテストを実行 (`bash scripts/medical_BTCV_SwinUNETR_quick.sh`)

---

## 📞 サポート

問題が解決しない場合:
1. このガイドのトラブルシューティングを確認
2. DATASET_SETUP.md の詳細手順を確認
3. エラーメッセージを確認して対処

**準備が整ったら実行してください！** 🚀
