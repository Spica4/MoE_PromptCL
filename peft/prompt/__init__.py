"""
Prompt modules for continual learning

Available prompt implementations:
- NoRGa Prompt (norga_prompt.py): Non-linear Residual Gates prompt
- HiDe Prompt (hide_prompt.py): Hierarchical Decomposition prompt
- DualPrompt/L2P (dp_prompt.py): DualPrompt and Learning to Prompt
- Swin UNETR Prompt (swin_unetr_prompt.py): 3D medical imaging prompt
"""

__all__ = ['norga_prompt', 'hide_prompt', 'dp_prompt', 'swin_unetr_prompt']
