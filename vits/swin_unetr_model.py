"""
Swin UNETR model wrapper for continual learning with NoRGa prompts

Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
Paper: https://arxiv.org/abs/2201.01266
MONAI implementation: https://docs.monai.io/en/stable/networks.html#swin-unetr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
import logging

_logger = logging.getLogger(__name__)


class SwinUNETRWithPrompt(nn.Module):
    """
    Swin UNETR model with prompt-based continual learning support

    Integrates MONAI's Swin UNETR with NoRGa prompt mechanism for
    continual learning on medical image segmentation tasks
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=14,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        # Prompt parameters
        use_e_prompt=True,
        e_prompt_length=10,
        e_prompt_layer_idx=[0, 1, 2, 3],
        prompt_pool=True,
        prompt_pool_size=5,
        prompt_key_dim=768,
        top_k=1,
        batchwise_prompt=True,
        prompt_key_init='uniform',
        embedding_key='mean',
        pull_constraint=True,
        pull_constraint_coeff=0.5,
        use_norga=True,
        gate_act='tanh',
        **kwargs
    ):
        """
        Args:
            img_size: Input image size (D, H, W)
            in_channels: Number of input channels (1 for CT/MRI)
            out_channels: Number of output segmentation classes
            feature_size: Base feature size for Swin transformer
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Stochastic depth rate
            use_checkpoint: Use gradient checkpointing to save memory
            spatial_dims: Spatial dimensions (3 for 3D)
            use_e_prompt: Whether to use E-Prompt
            e_prompt_length: Length of E-Prompt tokens
            e_prompt_layer_idx: Layers to insert prompts
            prompt_pool: Whether to use prompt pool
            prompt_pool_size: Size of prompt pool (number of tasks)
            prompt_key_dim: Dimension of prompt keys
            top_k: Number of prompts to select
            batchwise_prompt: Whether to use batchwise prompt selection
            prompt_key_init: Initialization method for prompt keys
            embedding_key: Key type for prompt selection ('mean', 'max', 'cls')
            pull_constraint: Whether to use pull constraint loss
            pull_constraint_coeff: Coefficient for pull constraint
            use_norga: Whether to use NoRGa gates
            gate_act: Activation function for gates
        """
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.spatial_dims = spatial_dims

        # Prompt parameters
        self.use_e_prompt = use_e_prompt
        self.e_prompt_length = e_prompt_length
        self.e_prompt_layer_idx = e_prompt_layer_idx
        self.prompt_pool = prompt_pool
        self.prompt_pool_size = prompt_pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.embedding_key = embedding_key
        self.pull_constraint = pull_constraint
        self.pull_constraint_coeff = pull_constraint_coeff
        self.use_norga = use_norga
        self.gate_act = gate_act

        # Create base Swin UNETR model
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        # Initialize prompt modules if using prompts
        if self.use_e_prompt and self.prompt_pool:
            self._init_prompt_modules()

        _logger.info(f"Created Swin UNETR with prompts: "
                    f"img_size={img_size}, in_channels={in_channels}, "
                    f"out_channels={out_channels}, feature_size={feature_size}")

    def _init_prompt_modules(self):
        """Initialize prompt-related modules"""
        # Get encoder hidden dimension
        # Swin UNETR encoder outputs feature_size * 8 dimension
        self.prompt_key_dim = self.feature_size * 16  # 768 for feature_size=48

        # Initialize prompt pool
        # Shape: (pool_size, length, embed_dim)
        self.e_prompt_pool = nn.Parameter(
            torch.randn(
                self.prompt_pool_size,
                self.e_prompt_length,
                self.prompt_key_dim
            )
        )
        nn.init.uniform_(self.e_prompt_pool, -1, 1)

        # Initialize prompt keys for selection
        self.prompt_key = nn.Parameter(
            torch.randn(
                self.prompt_pool_size,
                self.prompt_key_dim
            )
        )
        if self.prompt_key_init == 'uniform':
            nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            nn.init.normal_(self.prompt_key)

        # NoRGa gates for each prompt layer
        if self.use_norga:
            self.prompt_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.prompt_key_dim, self.prompt_key_dim),
                    nn.Tanh() if self.gate_act == 'tanh' else nn.Sigmoid()
                )
                for _ in self.e_prompt_layer_idx
            ])

    def forward_encoder_with_prompts(self, x, task_id=None, cls_features=None, train=False):
        """
        Forward pass through encoder with prompt integration

        Args:
            x: Input tensor (B, C, D, H, W)
            task_id: Task ID for prompt selection
            cls_features: Pre-computed features for prompt selection
            train: Whether in training mode

        Returns:
            Encoder hidden states at different scales
        """
        # Get encoder features from Swin UNETR
        hidden_states_out = self.swin_unetr.swinViT(x, normalize=True)

        # hidden_states_out contains features at 5 scales:
        # [scale 0, scale 1, scale 2, scale 3, scale 4]

        if self.use_e_prompt and self.prompt_pool:
            # Get feature for prompt selection
            if cls_features is None:
                # Use mean pooling of the deepest features for prompt key
                cls_features = hidden_states_out[-1].mean(dim=1)  # (B, D)

            # Select prompts
            prompt_mask, selected_prompts, reduce_sim = self._select_prompts(
                cls_features, task_id, train
            )

            # Apply prompts to specified layers
            for layer_idx in self.e_prompt_layer_idx:
                if layer_idx < len(hidden_states_out):
                    hidden_states_out[layer_idx] = self._apply_prompt_to_features(
                        hidden_states_out[layer_idx],
                        selected_prompts,
                        layer_idx
                    )

            return hidden_states_out, prompt_mask, reduce_sim

        return hidden_states_out, None, None

    def _select_prompts(self, cls_features, task_id, train):
        """
        Select prompts from pool based on input features

        Args:
            cls_features: Features for prompt selection (B, D)
            task_id: Task ID (if provided, use task-specific prompts)
            train: Whether in training mode

        Returns:
            prompt_mask: Mask indicating selected prompts
            selected_prompts: Selected prompt embeddings
            reduce_sim: Similarity scores for pull constraint
        """
        B = cls_features.shape[0]

        if task_id is not None:
            # Use task-specific prompts
            prompt_mask = torch.zeros(B, self.prompt_pool_size, device=cls_features.device)
            prompt_mask[:, task_id] = 1.0
            selected_prompts = self.e_prompt_pool[task_id:task_id+1].expand(B, -1, -1)
            reduce_sim = None
        else:
            # Compute similarity between features and prompt keys
            # cls_features: (B, D), prompt_key: (pool_size, D)
            cls_features_norm = F.normalize(cls_features, dim=1)
            prompt_key_norm = F.normalize(self.prompt_key, dim=1)

            # Similarity: (B, pool_size)
            similarity = torch.matmul(cls_features_norm, prompt_key_norm.t())

            # Select top-k prompts
            if self.top_k == 1:
                _, idx = torch.topk(similarity, k=1, dim=1)  # (B, 1)
                prompt_mask = F.one_hot(idx.squeeze(1), num_classes=self.prompt_pool_size).float()
                selected_prompts = self.e_prompt_pool[idx.squeeze(1)]  # (B, length, D)
            else:
                # Top-k selection
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # (B, k)
                prompt_mask = torch.zeros(B, self.prompt_pool_size, device=cls_features.device)
                prompt_mask.scatter_(1, idx, 1.0)

                # Average top-k prompts
                selected_prompts = torch.stack([
                    self.e_prompt_pool[idx[i]].mean(dim=0) for i in range(B)
                ])  # (B, length, D)

            # Compute pull constraint loss term
            if self.pull_constraint and train:
                # Get selected prompt keys
                selected_keys = self.prompt_key[idx.squeeze(1)]  # (B, D)
                reduce_sim = (1 - F.cosine_similarity(cls_features, selected_keys, dim=1)).mean()
            else:
                reduce_sim = None

        return prompt_mask, selected_prompts, reduce_sim

    def _apply_prompt_to_features(self, features, prompts, layer_idx):
        """
        Apply prompts to features with optional NoRGa gates

        Args:
            features: Feature tensor
            prompts: Prompt embeddings (B, length, D)
            layer_idx: Layer index

        Returns:
            Modified features
        """
        B = features.shape[0]

        # Apply NoRGa gate if enabled
        if self.use_norga:
            gate_idx = self.e_prompt_layer_idx.index(layer_idx)
            gate = self.prompt_gates[gate_idx]

            # Compute gate values
            gate_values = gate(prompts)  # (B, length, D)
            prompts = prompts * gate_values

        # Concatenate prompts with features
        # Note: For 3D features, prompts are added as additional spatial tokens
        # This is a simplified implementation; more sophisticated integration may be needed

        return features

    def forward(self, x, task_id=None, cls_features=None, train=False):
        """
        Forward pass

        Args:
            x: Input tensor (B, C, D, H, W)
            task_id: Task ID for task-specific prompts
            cls_features: Pre-computed features for prompt selection
            train: Whether in training mode

        Returns:
            logits: Segmentation predictions (B, num_classes, D, H, W)
            prompt_loss: Prompt-related losses (dict)
        """
        # Standard Swin UNETR forward if not using prompts
        if not self.use_e_prompt or not self.prompt_pool:
            logits = self.swin_unetr(x)
            return logits, {}

        # Forward with prompts (simplified - uses standard encoder)
        # Full integration would require modifying Swin encoder internals
        logits = self.swin_unetr(x)

        # TODO: Integrate prompt selection and application
        # This is a placeholder - full integration requires modifying MONAI's SwinUNETR
        prompt_loss = {}

        return logits, prompt_loss

    def freeze_backbone(self):
        """Freeze the Swin UNETR backbone"""
        for param in self.swin_unetr.parameters():
            param.requires_grad = False

        # Unfreeze prompt parameters
        if self.use_e_prompt and self.prompt_pool:
            self.e_prompt_pool.requires_grad = True
            self.prompt_key.requires_grad = True
            if self.use_norga:
                for gate in self.prompt_gates:
                    for param in gate.parameters():
                        param.requires_grad = True

    def get_prompt_params(self):
        """Get prompt-related parameters for separate optimization"""
        params = []
        if self.use_e_prompt and self.prompt_pool:
            params.append(self.e_prompt_pool)
            params.append(self.prompt_key)
            if self.use_norga:
                for gate in self.prompt_gates:
                    params.extend(gate.parameters())
        return params


def create_swin_unetr_model(args):
    """
    Create Swin UNETR model from config args

    Args:
        args: Configuration arguments

    Returns:
        SwinUNETRWithPrompt model
    """
    model = SwinUNETRWithPrompt(
        img_size=args.img_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=args.spatial_dims,
        use_e_prompt=args.use_e_prompt,
        e_prompt_length=args.length,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        prompt_pool=args.prompt_pool,
        prompt_pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        embedding_key=args.embedding_key,
        pull_constraint=args.pull_constraint,
        pull_constraint_coeff=args.pull_constraint_coeff,
        use_norga=args.use_norga if hasattr(args, 'use_norga') else True,
        gate_act=args.gate_act if hasattr(args, 'gate_act') else 'tanh',
    )

    # Load pretrained weights if specified
    if args.pretrained and args.pretrained_path:
        _logger.info(f"Loading pretrained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.swin_unetr.load_state_dict(checkpoint, strict=False)

    return model


if __name__ == "__main__":
    # Test model creation
    import argparse

    args = argparse.Namespace(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=14,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        use_e_prompt=True,
        length=10,
        e_prompt_layer_idx=[0, 1, 2, 3],
        prompt_pool=True,
        size=5,
        top_k=1,
        batchwise_prompt=True,
        prompt_key_init='uniform',
        embedding_key='mean',
        pull_constraint=True,
        pull_constraint_coeff=0.5,
        pretrained=False,
        pretrained_path='',
    )

    model = create_swin_unetr_model(args)
    print(f"Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        out, losses = model(x, task_id=0)
    print(f"Output shape: {out.shape}")
