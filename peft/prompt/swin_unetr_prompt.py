"""
3D Prompt module for Swin UNETR with NoRGa gates
Adapted from NoRGa prompt for 3D medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EPrompt3D(nn.Module):
    """
    3D E-Prompt module for continual learning on medical image segmentation

    Supports:
    - Prompt pooling for multiple tasks
    - NoRGa (non-linear residual gates)
    - 3D spatial feature integration
    """

    def __init__(
        self,
        length=10,
        embed_dim=768,
        embedding_key='mean',
        prompt_init='uniform',
        prompt_pool=True,
        prompt_key=True,
        pool_size=5,
        top_k=1,
        batchwise_prompt=False,
        prompt_key_init='uniform',
        num_layers=1,
        use_prefix_tune_for_e_prompt=False,
        num_heads=12,
        same_key_value=False,
        gate_act='tanh',
        use_norga=True,
    ):
        """
        Args:
            length: Length of prompt tokens
            embed_dim: Embedding dimension
            embedding_key: Key type for prompt selection ('mean', 'max', 'cls')
            prompt_init: Prompt initialization ('uniform', 'zero')
            prompt_pool: Whether to use prompt pool
            prompt_key: Whether to use learnable prompt keys
            pool_size: Size of prompt pool (number of tasks)
            top_k: Number of prompts to select
            batchwise_prompt: Whether to use batchwise prompt selection
            prompt_key_init: Prompt key initialization
            num_layers: Number of layers with prompts
            use_prefix_tune_for_e_prompt: Whether to use prefix tuning
            num_heads: Number of attention heads (for prefix tuning)
            same_key_value: Whether to use same key and value
            gate_act: Gate activation function ('tanh', 'sigmoid')
            use_norga: Whether to use NoRGa gates
        """
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.gate_act = gate_act
        self.use_norga = use_norga
        self.embed_dim = embed_dim

        # Initialize prompt pool
        if self.prompt_pool:
            self._init_prompt_pool()

        # Initialize prompt keys
        if prompt_key:
            self._init_prompt_keys()

        # Initialize NoRGa gates
        if self.use_norga:
            self._init_norga_gates()

    def _init_prompt_pool(self):
        """Initialize prompt pool"""
        if self.use_prefix_tune_for_e_prompt:
            # Prefix tuning style (for attention layers)
            assert self.embed_dim % self.num_heads == 0

            if self.same_key_value:
                prompt_pool_shape = (
                    self.num_layers, 1, self.pool_size, self.length,
                    self.num_heads, self.embed_dim // self.num_heads
                )
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
            else:
                prompt_pool_shape = (
                    self.num_layers, 2, self.pool_size, self.length,
                    self.num_heads, self.embed_dim // self.num_heads
                )
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
        else:
            # Standard prompt style
            prompt_pool_shape = (self.num_layers, self.pool_size, self.length, self.embed_dim)
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

    def _init_prompt_keys(self):
        """Initialize learnable prompt keys"""
        key_shape = (self.pool_size, self.embed_dim)
        if self.prompt_init == 'zero':
            self.prompt_key_param = nn.Parameter(torch.zeros(key_shape))
        elif self.prompt_init == 'uniform':
            self.prompt_key_param = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key_param, -1, 1)

    def _init_norga_gates(self):
        """Initialize NoRGa gates for each layer"""
        self.gates = nn.ModuleList()

        for _ in range(self.num_layers):
            if self.use_prefix_tune_for_e_prompt:
                # Gate for prefix tuning (operates on multi-head features)
                gate = nn.Sequential(
                    nn.Linear(self.embed_dim // self.num_heads, self.embed_dim // self.num_heads),
                    nn.Tanh() if self.gate_act == 'tanh' else nn.Sigmoid()
                )
            else:
                # Gate for standard prompts
                gate = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Tanh() if self.gate_act == 'tanh' else nn.Sigmoid()
                )
            self.gates.append(gate)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix"""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def select_prompts(self, x_embed, task_id=None):
        """
        Select prompts based on input features or task_id

        Args:
            x_embed: Input features for prompt selection (B, D) or (B, L, D)
            task_id: Task ID for task-specific prompts

        Returns:
            prompt_idx: Selected prompt indices (B,) or (B, top_k)
            prompt_weight: Prompt selection weights (if using soft selection)
        """
        B = x_embed.shape[0]

        if task_id is not None:
            # Use task-specific prompts
            prompt_idx = torch.full((B,), task_id, dtype=torch.long, device=x_embed.device)
            prompt_weight = None
        else:
            # Compute features for prompt selection
            if len(x_embed.shape) == 3:  # (B, L, D)
                if self.embedding_key == 'mean':
                    key_features = x_embed.mean(dim=1)  # (B, D)
                elif self.embedding_key == 'max':
                    key_features = x_embed.max(dim=1)[0]  # (B, D)
                elif self.embedding_key == 'cls':
                    key_features = x_embed[:, 0]  # (B, D)
                else:
                    key_features = x_embed.mean(dim=1)
            else:  # (B, D)
                key_features = x_embed

            # Normalize features and keys
            key_features_norm = self.l2_normalize(key_features, dim=1)
            prompt_key_norm = self.l2_normalize(self.prompt_key_param, dim=1)

            # Compute similarity
            similarity = torch.matmul(key_features_norm, prompt_key_norm.t())  # (B, pool_size)

            # Select top-k prompts
            if self.top_k == 1:
                _, prompt_idx = torch.topk(similarity, k=1, dim=1)
                prompt_idx = prompt_idx.squeeze(1)  # (B,)
                prompt_weight = None
            else:
                _, prompt_idx = torch.topk(similarity, k=self.top_k, dim=1)  # (B, top_k)
                # Use top-k similarities as weights
                prompt_weight = F.softmax(torch.gather(similarity, 1, prompt_idx), dim=1)  # (B, top_k)

        return prompt_idx, prompt_weight

    def forward(self, x_embed, prompt_mask=None, prompt_idx=None, prompt_weight=None, layer_idx=0):
        """
        Forward pass to get prompted features

        Args:
            x_embed: Input embeddings (not used for prompt retrieval, kept for compatibility)
            prompt_mask: Binary mask for prompt selection (B, pool_size)
            prompt_idx: Prompt indices (B,) or (B, top_k)
            prompt_weight: Prompt weights for soft selection (B, top_k)
            layer_idx: Current layer index

        Returns:
            Dictionary containing:
                - batched_prompt: Selected prompts with optional gates
                - prompt_idx: Prompt indices used
        """
        out = {}

        if not self.prompt_pool:
            return out

        # Get prompt indices
        if prompt_mask is not None:
            # Convert mask to indices
            prompt_idx = torch.argmax(prompt_mask, dim=1)  # (B,)
        elif prompt_idx is None:
            raise ValueError("Either prompt_mask or prompt_idx must be provided")

        out['prompt_idx'] = prompt_idx

        # Retrieve prompts
        if self.use_prefix_tune_for_e_prompt:
            # Prefix tuning style
            if prompt_weight is not None:
                # Soft prompt selection
                batched_prompt_raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight, self.prompt[layer_idx:layer_idx+1])
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                # Hard prompt selection
                if len(prompt_idx.shape) == 1:  # (B,)
                    batched_prompt_raw = self.prompt[layer_idx:layer_idx+1, :, prompt_idx]  # (1, 2, B, length, heads, head_dim)
                else:  # (B, top_k)
                    batched_prompt_raw = self.prompt[layer_idx:layer_idx+1, :, prompt_idx]  # (1, 2, B, top_k, length, heads, head_dim)

                num_layers, dual, batch_size, *rest = batched_prompt_raw.shape
                if len(rest) == 3:  # Single prompt per batch
                    length, num_heads, heads_embed_dim = rest
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, length, num_heads, heads_embed_dim
                    )
                else:  # Multiple prompts per batch
                    top_k, length, num_heads, heads_embed_dim = rest
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )

            # Apply NoRGa gates
            if self.use_norga:
                gate = self.gates[layer_idx]
                # Apply gate to each head separately
                gate_values = gate(batched_prompt)  # Same shape as batched_prompt
                batched_prompt = batched_prompt * gate_values

        else:
            # Standard prompt style
            if prompt_weight is not None:
                # Soft prompt selection
                batched_prompt_raw = torch.einsum("bp,npld->nbpld", prompt_weight, self.prompt[layer_idx:layer_idx+1])
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )
            else:
                # Hard prompt selection
                if len(prompt_idx.shape) == 1:  # (B,)
                    batched_prompt_raw = self.prompt[layer_idx:layer_idx+1, prompt_idx]  # (1, B, length, D)
                else:  # (B, top_k)
                    batched_prompt_raw = self.prompt[layer_idx:layer_idx+1, prompt_idx]  # (1, B, top_k, length, D)

                num_layers, batch_size, *rest = batched_prompt_raw.shape
                if len(rest) == 2:  # Single prompt per batch
                    length, embed_dim = rest
                    batched_prompt = batched_prompt_raw  # Keep as is
                else:  # Multiple prompts per batch
                    top_k, length, embed_dim = rest
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )

            # Apply NoRGa gates
            if self.use_norga:
                gate = self.gates[layer_idx]
                gate_values = gate(batched_prompt)
                batched_prompt = batched_prompt * gate_values

        out['batched_prompt'] = batched_prompt.squeeze(0)  # Remove num_layers dim

        return out


def create_prompt_module_3d(args):
    """
    Create 3D prompt module from config args

    Args:
        args: Configuration arguments

    Returns:
        EPrompt3D module
    """
    prompt_module = EPrompt3D(
        length=args.length,
        embed_dim=args.feature_size * 16,  # Swin UNETR embedding dimension
        embedding_key=args.embedding_key,
        prompt_init='uniform',
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        num_layers=len(args.e_prompt_layer_idx),
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        num_heads=12,  # Default for Swin Transformer
        same_key_value=False,
        gate_act=args.gate_act if hasattr(args, 'gate_act') else 'tanh',
        use_norga=args.use_norga if hasattr(args, 'use_norga') else True,
    )

    return prompt_module


if __name__ == "__main__":
    # Test prompt module
    prompt_module = EPrompt3D(
        length=10,
        embed_dim=768,
        prompt_pool=True,
        pool_size=5,
        top_k=1,
        num_layers=4,
        use_norga=True,
        gate_act='tanh',
    )

    print(f"Prompt module created successfully")
    print(f"Total parameters: {sum(p.numel() for p in prompt_module.parameters()):,}")

    # Test forward pass
    B, L, D = 2, 100, 768
    x_embed = torch.randn(B, L, D)

    # Select prompts
    prompt_idx, prompt_weight = prompt_module.select_prompts(x_embed)
    print(f"Selected prompt indices: {prompt_idx}")

    # Get batched prompts for layer 0
    out = prompt_module.forward(x_embed, prompt_idx=prompt_idx, layer_idx=0)
    print(f"Batched prompt shape: {out['batched_prompt'].shape}")
