"""Low-Rank Adaptation (LoRA) Parameter-Efficient Fine-Tuning (PEFT) module.

Enables efficient fine-tuning of Geospatial Foundation Models and Vision Transformers
by freezing base backbone parameters and training low-rank decomposition matrices
on Multi-Head Attention query and value projections (W_q, W_v).
"""

import math
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping standard linear projection layers."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)

        # Freeze the base layer
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA trainable low-rank adapters: W_delta = (B * A) * (alpha / r)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.merged = False
        self.reset_parameters()

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def reset_parameters(self):
        """Initialize LoRA weights: A with Kaiming uniform, B to zero."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge low-rank weights into the base weight for zero-latency inference."""
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerge low-rank weights back to separate parameters."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        base_out = self.base_layer(x)
        adapted = self.dropout(x)
        adapted_out = (adapted @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + adapted_out


def freeze_model_parameters(model: nn.Module) -> None:
    """Freeze all parameters in the given model."""
    for param in model.parameters():
        param.requires_grad = False


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_names: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, int]]:
    """Inject LoRA adapters into target attention projection layers and freeze base model.

    Args:
        model: PyTorch model containing Vision Transformer or attention layers.
        rank: Decomposition rank.
        alpha: Scaling factor.
        target_names: Target projection names to adapt (default: q_proj and v_proj).

    Returns:
        Adapted model and dictionary of parameter statistics.
    """
    if target_names is None:
        target_names = ["q_proj", "v_proj"]

    # 1. Freeze entire base model
    freeze_model_parameters(model)

    adapted_count = 0
    # 2. Find and replace matching nn.Linear modules
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_child_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(tgt in child_name for tgt in target_names):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                adapted_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "adapted_layers": adapted_count,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_fraction": trainable_params / max(1, total_params),
    }
    return model, stats


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only trainable LoRA parameters for lightweight checkpointing."""
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    """Load saved LoRA parameters into adapted model."""
    model.load_state_dict(lora_state, strict=False)
