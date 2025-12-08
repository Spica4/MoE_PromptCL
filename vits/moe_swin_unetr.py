"""
MoE-based Swin UNETR for 3D Medical Image Segmentation with Continual Learning
Based on NoRGa (Non-linear Residual Gates) from "Mixture of Experts Meets Prompt-Based Continual Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
import logging

_logger = logging.getLogger(__name__)


class MedicalExpertPrompt(nn.Module):
    """
    Expert prompts for 3D medical image segmentation
    Each task has its own set of expert prompts
    """
    def __init__(
        self,
        num_tasks=4,
        num_experts_per_task=4,
        prompt_length=5,
        embed_dim=768,
        num_layers=1,
        prompt_init='uniform',
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_experts_per_task = num_experts_per_task
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Total expert pool size
        self.pool_size = num_tasks * num_experts_per_task

        # Expert prompt pool: [num_layers, pool_size, prompt_length, embed_dim]
        prompt_pool_shape = (num_layers, self.pool_size, prompt_length, embed_dim)

        if prompt_init == 'zero':
            self.expert_prompts = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == 'uniform':
            self.expert_prompts = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.expert_prompts, -1, 1)
        else:
            self.expert_prompts = nn.Parameter(torch.randn(prompt_pool_shape))

        # Task-to-experts mapping
        self.task_expert_mapping = {}
        for task_id in range(num_tasks):
            start_idx = task_id * num_experts_per_task
            end_idx = start_idx + num_experts_per_task
            self.task_expert_mapping[task_id] = list(range(start_idx, end_idx))

        _logger.info(f"Created MedicalExpertPrompt: {num_tasks} tasks, "
                    f"{num_experts_per_task} experts/task, pool_size={self.pool_size}")

    def get_task_experts(self, task_id):
        """
        Get expert indices for a specific task

        Args:
            task_id: Task ID (0-indexed)

        Returns:
            List of expert indices for this task
        """
        if task_id >= self.num_tasks:
            _logger.warning(f"Task ID {task_id} exceeds num_tasks {self.num_tasks}")
            task_id = self.num_tasks - 1

        return self.task_expert_mapping[task_id]

    def forward(self, task_id, expert_weights=None):
        """
        Select and combine expert prompts for a task

        Args:
            task_id: Current task ID
            expert_weights: Optional weights for expert combination [batch_size, num_experts]
                          If None, use equal weights

        Returns:
            Combined expert prompts [num_layers, batch_size, prompt_length, embed_dim]
        """
        expert_indices = self.get_task_experts(task_id)

        # Get experts for this task: [num_layers, num_experts, prompt_length, embed_dim]
        task_experts = self.expert_prompts[:, expert_indices]

        if expert_weights is None:
            # Equal weight combination
            combined_prompts = task_experts.mean(dim=1, keepdim=True)  # [num_layers, 1, prompt_length, embed_dim]
        else:
            # Weighted combination: [num_layers, batch_size, prompt_length, embed_dim]
            # expert_weights: [batch_size, num_experts]
            combined_prompts = torch.einsum('be,neld->nbld', expert_weights, task_experts)

        return combined_prompts


class GatingNetwork(nn.Module):
    """
    Gating network to compute expert weights based on input features
    Implements NoRGa-style non-linear residual gating
    """
    def __init__(
        self,
        input_dim,
        num_experts,
        hidden_dim=256,
        use_nonlinear=True,
        use_residual=True,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.use_nonlinear = use_nonlinear
        self.use_residual = use_residual

        # Linear gating
        self.gate_linear = nn.Linear(input_dim, num_experts)

        if use_nonlinear:
            # Non-linear gating path
            self.gate_nonlinear = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts),
            )

        # Scaling factors for residual connection
        if use_residual:
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Compute gating weights

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            Expert weights [batch_size, num_experts]
        """
        # Linear gating scores
        linear_scores = self.gate_linear(x)

        if self.use_nonlinear and self.use_residual:
            # NoRGa: non-linear residual gating
            nonlinear_scores = self.gate_nonlinear(x)
            # Residual connection with scaling
            scores = linear_scores + self.alpha * torch.relu(linear_scores * self.beta) + nonlinear_scores
        elif self.use_nonlinear:
            # Non-linear only
            nonlinear_scores = self.gate_nonlinear(x)
            scores = linear_scores + nonlinear_scores
        else:
            # Linear only
            scores = linear_scores

        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)

        return weights


class MoESwinUNETR(nn.Module):
    """
    Swin UNETR with Mixture of Experts for continual learning
    """
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=16,
        feature_size=48,
        num_tasks=4,
        num_experts_per_task=4,
        prompt_length=5,
        use_moe=True,
        use_nonlinear_gate=True,
        use_residual_gate=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
    ):
        """
        Args:
            img_size: Input image size (D, H, W)
            in_channels: Number of input channels
            out_channels: Number of output classes
            feature_size: Base feature size for Swin transformer
            num_tasks: Number of continual learning tasks
            num_experts_per_task: Number of experts per task
            prompt_length: Length of expert prompts
            use_moe: Whether to use MoE mechanism
            use_nonlinear_gate: Use non-linear gating
            use_residual_gate: Use residual connection in gating
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Stochastic depth rate
            use_checkpoint: Use gradient checkpointing
            spatial_dims: Spatial dimensions (3 for 3D)
        """
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.spatial_dims = spatial_dims
        self.num_tasks = num_tasks
        self.use_moe = use_moe

        # Base Swin UNETR model
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

        if use_moe:
            # Expert prompts
            # Use feature_size * 16 as embed_dim (typical Swin UNETR encoder dim)
            encoder_dim = feature_size * 16  # 768 for feature_size=48

            self.expert_prompts = MedicalExpertPrompt(
                num_tasks=num_tasks,
                num_experts_per_task=num_experts_per_task,
                prompt_length=prompt_length,
                embed_dim=encoder_dim,
                num_layers=1,
                prompt_init='uniform',
            )

            # Gating network
            # Input: global pooled features from encoder
            self.gating_network = GatingNetwork(
                input_dim=encoder_dim,
                num_experts=num_experts_per_task,
                hidden_dim=encoder_dim // 2,
                use_nonlinear=use_nonlinear_gate,
                use_residual=use_residual_gate,
            )

            # Prompt injection layer (adds prompts to encoder features)
            self.prompt_proj = nn.Linear(encoder_dim, encoder_dim)

        _logger.info(f"Created MoESwinUNETR: img_size={img_size}, "
                    f"out_channels={out_channels}, use_moe={use_moe}")

    def forward(self, x, task_id=0, train=False):
        """
        Forward pass

        Args:
            x: Input tensor [B, C, D, H, W]
            task_id: Current task ID
            train: Whether in training mode

        Returns:
            logits: Segmentation predictions [B, num_classes, D, H, W]
            aux_outputs: Dictionary with auxiliary outputs (e.g., expert weights)
        """
        aux_outputs = {}

        if self.use_moe and train:
            # Get encoder features for gating
            # Use the Swin UNETR encoder to extract features
            hidden_states_out = self.swin_unetr.swinViT(x, normalize=True)
            encoder_features = hidden_states_out[4]  # Deepest features

            # Global average pooling for gating
            # [B, C, D, H, W] -> [B, C]
            pooled_features = F.adaptive_avg_pool3d(encoder_features, 1).view(encoder_features.size(0), -1)

            # Compute expert weights via gating network
            expert_weights = self.gating_network(pooled_features)  # [B, num_experts]
            aux_outputs['expert_weights'] = expert_weights

            # Get combined expert prompts
            expert_prompts = self.expert_prompts(task_id, expert_weights)  # [num_layers, B, prompt_length, embed_dim]

            # Inject prompts into features (simplified - adds to spatial features)
            # In practice, this would be integrated into attention layers
            # For now, we add prompt information via projection
            prompt_features = expert_prompts.mean(dim=[0, 2])  # [B, embed_dim]
            prompt_features = self.prompt_proj(prompt_features).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Add prompt features to encoder output (residual)
            encoder_features = encoder_features + prompt_features

            # Continue with decoder using modified features
            # Note: This is a simplified integration. Full integration would modify
            # the Swin UNETR architecture to include prompts in attention layers
            logits = self.swin_unetr.out(self.swin_unetr.decoder5(
                self.swin_unetr.decoder4(
                    self.swin_unetr.decoder3(
                        self.swin_unetr.decoder2(
                            self.swin_unetr.decoder1(encoder_features, hidden_states_out[3]),
                            hidden_states_out[2]
                        ),
                        hidden_states_out[1]
                    ),
                    hidden_states_out[0]
                ),
                self.swin_unetr.encoder1(x)
            ))
        else:
            # Standard forward without MoE
            logits = self.swin_unetr(x)

        return logits, aux_outputs

    def expand_output_layer(self, new_out_channels):
        """
        Expand output layer for class-incremental learning

        Args:
            new_out_channels: New number of output channels
        """
        if new_out_channels <= self.out_channels:
            _logger.info(f"Output channels already {self.out_channels}, no expansion needed")
            return

        _logger.info(f"Expanding output layer from {self.out_channels} to {new_out_channels} channels")

        # Get the current output layer
        old_out_layer = self.swin_unetr.out
        old_weight = old_out_layer.conv.conv.weight.data
        old_bias = old_out_layer.conv.conv.bias.data if old_out_layer.conv.conv.bias is not None else None

        # Create new output layer with more channels
        from monai.networks.blocks import UnetOutBlock
        new_out_layer = UnetOutBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feature_size,
            out_channels=new_out_channels,
        )

        # Initialize new layer
        # Copy old weights for existing channels
        with torch.no_grad():
            new_out_layer.conv.conv.weight[:self.out_channels] = old_weight
            if old_bias is not None:
                new_out_layer.conv.conv.bias[:self.out_channels] = old_bias
            # Initialize new channels with small random values
            nn.init.kaiming_normal_(new_out_layer.conv.conv.weight[self.out_channels:], mode='fan_out')
            if new_out_layer.conv.conv.bias is not None:
                nn.init.constant_(new_out_layer.conv.conv.bias[self.out_channels:], 0)

        # Replace output layer
        self.swin_unetr.out = new_out_layer
        self.out_channels = new_out_channels

        _logger.info(f"Output layer expanded successfully")


def create_moe_swin_unetr_model(args):
    """
    Create MoE Swin UNETR model from args

    Args:
        args: Configuration arguments

    Returns:
        MoE Swin UNETR model
    """
    # Convert list to tuple for MONAI compatibility
    img_size = tuple(args.img_size) if isinstance(args.img_size, list) else args.img_size

    model = MoESwinUNETR(
        img_size=img_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        num_tasks=getattr(args, 'num_tasks', 4),
        num_experts_per_task=getattr(args, 'num_experts_per_task', 4),
        prompt_length=getattr(args, 'prompt_length', 5),
        use_moe=getattr(args, 'use_moe', True),
        use_nonlinear_gate=getattr(args, 'use_nonlinear_gate', True),
        use_residual_gate=getattr(args, 'use_residual_gate', True),
        drop_rate=getattr(args, 'drop_rate', 0.0),
        attn_drop_rate=getattr(args, 'attn_drop_rate', 0.0),
        dropout_path_rate=getattr(args, 'dropout_path_rate', 0.0),
        use_checkpoint=getattr(args, 'use_checkpoint', False),
        spatial_dims=getattr(args, 'spatial_dims', 3),
    )

    return model
