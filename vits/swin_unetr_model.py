"""
Swin UNETR Model Wrapper for 3D Medical Image Segmentation
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import logging

_logger = logging.getLogger(__name__)


class SwinUNETRWrapper(nn.Module):
    """
    Wrapper for MONAI's Swin UNETR with continual learning support
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=16,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
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
            use_checkpoint: Use gradient checkpointing
            spatial_dims: Spatial dimensions (3 for 3D)
        """
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.spatial_dims = spatial_dims

        # Create Swin UNETR model
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

        _logger.info(f"Created Swin UNETR: img_size={img_size}, "
                    f"in_channels={in_channels}, out_channels={out_channels}")

    def forward(self, x, task_id=None, train=False):
        """
        Forward pass

        Args:
            x: Input tensor (B, C, D, H, W)
            task_id: Task ID (for continual learning, unused in basic version)
            train: Whether in training mode

        Returns:
            logits: Segmentation predictions (B, num_classes, D, H, W)
            prompt_loss: Empty dict (for compatibility)
        """
        logits = self.swin_unetr(x)
        return logits, {}

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

    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove prefixes
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('swinViT.', '')
            new_state_dict[name] = v

        self.swin_unetr.load_state_dict(new_state_dict, strict=False)
        _logger.info(f"Loaded pretrained weights from {checkpoint_path}")


def create_swin_unetr_model(args):
    """
    Create Swin UNETR model from args

    Args:
        args: Configuration arguments

    Returns:
        Swin UNETR model
    """
    # Convert list to tuple for MONAI compatibility
    img_size = tuple(args.img_size) if isinstance(args.img_size, list) else args.img_size

    model = SwinUNETRWrapper(
        img_size=img_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=getattr(args, 'drop_rate', 0.0),
        attn_drop_rate=getattr(args, 'attn_drop_rate', 0.0),
        dropout_path_rate=getattr(args, 'dropout_path_rate', 0.0),
        use_checkpoint=getattr(args, 'use_checkpoint', False),
        spatial_dims=getattr(args, 'spatial_dims', 3),
    )

    # Load pretrained weights if specified
    if getattr(args, 'pretrained', False) and getattr(args, 'pretrained_path', None):
        model.load_pretrained(args.pretrained_path)

    return model
