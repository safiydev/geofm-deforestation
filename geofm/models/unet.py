"""U-Net semantic segmentation architecture with ResNet and Vision Transformer backbones.

Designed for pixel-level canopy disturbance and logging road delineation in
high-resolution satellite imagery (Sentinel-2 and Planet NICFI).
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual convolutional block with GroupNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + res)


class ViTMultiHeadAttention(nn.Module):
    """Multi-Head Attention with explicit W_q, W_k, W_v projections for LoRA PEFT."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query

        b, n_q, _ = query.shape
        _, n_k, _ = key.shape

        q = self.q_proj(query).view(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, n_q, self.embed_dim)
        return self.out_proj(out)


class ViTBottleneck(nn.Module):
    """Vision Transformer self-attention bottleneck for global spatial context."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = ViTMultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W) where C == embed_dim.
        """
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        norm_seq = self.norm1(seq)
        attn_out = self.mha(norm_seq)
        seq = seq + attn_out
        seq = seq + self.mlp(self.norm2(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w)


class DecoderBlock(nn.Module):
    """U-Net decoder block with bilinear upsampling, skip concatenation, and residual convs."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x_up = torch.cat([x_up, skip], dim=1)
        h = self.act(self.gn1(self.conv1(x_up)))
        h = self.act(self.gn2(self.conv2(h)))
        return h


class UNet(nn.Module):
    """Full U-Net model with hybrid ResNet encoder and ViT bottleneck."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 1,
        features: Optional[List[int]] = None,
        use_vit_bottleneck: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.features = features
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, features[0]), num_channels=features[0]),
            nn.SiLU(inplace=True),
        )

        # Encoder pyramid
        self.enc1 = ResBlock(features[0], features[0])
        self.enc2 = ResBlock(features[0], features[1], stride=2)
        self.enc3 = ResBlock(features[1], features[2], stride=2)
        self.enc4 = ResBlock(features[2], features[3], stride=2)

        # Bottleneck
        if use_vit_bottleneck:
            self.bottleneck = ViTBottleneck(embed_dim=features[3], num_heads=4)
        else:
            self.bottleneck = nn.Identity()

        # Decoder pyramid
        self.dec3 = DecoderBlock(features[3], features[2], features[2])
        self.dec2 = DecoderBlock(features[2], features[1], features[1])
        self.dec1 = DecoderBlock(features[1], features[0], features[0])

        # Final classification head
        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature hierarchy for Siamese / fusion models."""
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        return [e1, e2, e3, b]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, b = self.extract_features(x)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        logits = self.head(d1)
        return logits
