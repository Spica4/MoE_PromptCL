# MoE Swin UNETR for 3D Medical Image Segmentation

## 概要

このドキュメントでは、NeurIPS 2024論文「Mixture of Experts Meets Prompt-Based Continual Learning」のアイデアを3D医療画像セグメンテーションに適用したMoE Swin UNETRの実装について説明します。

## アーキテクチャ

### 1. 基本構造

MoE Swin UNETRは、標準のSwin UNETRにMixture of Experts (MoE)メカニズムを統合したものです：

```
入力 (CT画像)
  ↓
Swin Transformer Encoder
  ↓
[MoE Layer]
  ├─ Expert Prompts (タスクごと)
  ├─ Gating Network (NoRGa)
  └─ Feature Modulation
  ↓
UNETR Decoder
  ↓
出力 (セグメンテーション)
```

### 2. 主要コンポーネント

#### 2.1 MedicalExpertPrompt

各タスクに対して複数のエキスパート（専門家）を割り当てます：

- **タスク数**: デフォルト4
- **タスクごとのexpert数**: デフォルト4
- **合計expert数**: 4 × 4 = 16

**例:**
```python
# Task 0: Expert 0, 1, 2, 3
# Task 1: Expert 4, 5, 6, 7
# Task 2: Expert 8, 9, 10, 11
# Task 3: Expert 12, 13, 14, 15
```

各expertは学習可能なプロンプト（特徴表現）です：
- **形状**: `[num_layers, pool_size, prompt_length, embed_dim]`
- **デフォルト**: `[1, 16, 5, 768]`

#### 2.2 GatingNetwork (NoRGa)

入力特徴に基づいてexpertの重みを計算します。

**NoRGa (Non-linear Residual Gates) の特徴:**

1. **線形ゲーティング**:
   ```python
   linear_scores = Linear(features)
   ```

2. **非線形ゲーティング**:
   ```python
   nonlinear_scores = ReLU(Linear(features))
   ```

3. **残差接続**:
   ```python
   final_scores = linear_scores + α * ReLU(linear_scores * β) + nonlinear_scores
   weights = Softmax(final_scores)
   ```

**利点:**
- より表現力の高いゲーティング
- タスク固有の特徴を動的に選択
- 勾配の流れを改善

#### 2.3 Expert Selection & Combination

ゲーティングネットワークから得られた重みでexpertを組み合わせます：

```python
combined_features = Σ(weight_i × expert_i)
```

### 3. 継続学習における役割

#### タスクごとの動作

**Task 0 (spleen)**:
- Expert 0-3を使用
- ゲーティングが最も関連性の高いexpertを選択
- 例: Expert 1(60%), Expert 2(30%), Expert 0(10%)

**Task 1 (right kidney)**:
- Expert 4-7を使用
- 新しいタスクに特化したexpertを学習
- 過去のexpert (0-3) は凍結可能

**利点:**
1. **カタストロフィック忘却の軽減**: 各タスクが独自のexpertセットを持つ
2. **タスク間の知識共有**: ゲーティングメカニズムが柔軟に組み合わせる
3. **パラメータ効率**: expertプロンプトは小さい（数千パラメータ）

## 実装の詳細

### ファイル構成

```
vits/
├── swin_unetr_model.py      # 標準Swin UNETR
└── moe_swin_unetr.py        # MoE版Swin UNETR (NEW)
    ├── MedicalExpertPrompt  # Expertプロンプト管理
    ├── GatingNetwork        # NoRGaゲーティング
    └── MoESwinUNETR         # メインモデル
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--use_moe` | False | MoEを使用 |
| `--num_experts_per_task` | 4 | タスクごとのexpert数 |
| `--prompt_length` | 5 | expertプロンプトの長さ |
| `--use_nonlinear_gate` | True | 非線形ゲーティング |
| `--use_residual_gate` | True | 残差接続ゲーティング (NoRGa) |

## 使用方法

### 1. MoE Swin UNETRでトレーニング

```bash
bash scripts/amos22_moe_swin_unetr_test.sh  # テスト実行
bash scripts/amos22_moe_swin_unetr.sh       # フル学習
```

### 2. 標準Swin UNETRとの比較

**標準版**:
```bash
python main.py swin_unetr \
--dataset AMOS22 \
--data-path /path/to/data \
--output_dir ./output/standard \
--num_tasks 4
```

**MoE版**:
```bash
python main.py swin_unetr \
--dataset AMOS22 \
--data-path /path/to/data \
--output_dir ./output/moe \
--num_tasks 4 \
--use_moe \
--num_experts_per_task 4
```

### 3. Expert重みの可視化

```python
# トレーニング中
logits, aux_outputs = model(x, task_id=1, train=True)
expert_weights = aux_outputs['expert_weights']  # [batch_size, num_experts]

# Expert 0が60%、Expert 1が30%、Expert 2が10%のように使用される
print(expert_weights[0])  # 例: [0.6, 0.3, 0.1, 0.0]
```

## 理論的背景

### Vision TransformerにおけるMoE

論文によると、Vision Transformerの注意機構は暗黙的にMoEをエンコードしています：

1. **Experts**: Key-Value ペア（各トークンが1つのexpert）
2. **Gating**: Attention スコア（Query-Key 相関）
3. **Output**: 重み付けされたValue の和

### NoRGaの利点

**標準ゲーティング** (線形):
```
gate(x) = Softmax(W·x)
```

**NoRGa** (非線形+残差):
```
gate(x) = Softmax(W·x + α·ReLU(W·x·β) + MLP(x))
```

**理論的利点:**
1. 表現力の向上（非線形変換）
2. 勾配の流れ改善（残差接続）
3. タスク固有の適応（追加のMLP）

## パフォーマンス期待値

### メモリ使用量

**追加パラメータ**:
- Expert prompts: `num_tasks × num_experts × prompt_length × embed_dim`
  - 例: 4 × 4 × 5 × 768 = 61,440 パラメータ
- Gating network: `~100,000` パラメータ

**合計追加**: 約0.16M パラメータ（標準Swin UNETRの~0.1%）

### 期待される効果

1. **忘却軽減**: タスク固有のexpertによりカタストロフィック忘却を軽減
2. **タスク適応**: ゲーティングメカニズムによる柔軟な適応
3. **知識転移**: Expert間の暗黙的な知識共有

## 今後の拡張

### 可能な改善

1. **Attention層への統合**: 現在は簡易版、Attention層に直接統合可能
2. **階層的Expert**: 異なるスケールのexpertを使用
3. **動的Expert追加**: タスクごとに動的にexpertを追加
4. **Expert蒸留**: 学習済みexpertから新しいexpertへ知識を転移

## 参考文献

- **論文**: "Mixture of Experts Meets Prompt-Based Continual Learning" (NeurIPS 2024)
- **ArXiv**: https://arxiv.org/abs/2405.14124
- **MONAI**: https://monai.io/
- **Swin UNETR**: "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis"

## トラブルシューティング

### Q: MoEを使うとメモリ不足になる
A: `--num_experts_per_task` を減らすか、`--batch-size` を小さくしてください

### Q: Expert重みが偏っている（1つのexpertに集中）
A: `--lr` を調整するか、expert初期化を変更してください

### Q: 標準版より性能が低い
A: エポック数を増やすか、`--prompt_length` を調整してください
