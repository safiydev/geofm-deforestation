"""Siamese Bitemporal Change Detection Network for Deforestation.

Extracts multi-scale deep features from pre-disturbance (T1) and post-disturbance (T2)
satellite acquisitions, computes differential feature embeddings, and decodes
pixel-accurate deforestation boundaries.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from geofm.models.unet import UNet, DecoderBlock


class DifferenceFusionBlock(nn.Module):
    """Combines T1 and T2 feature maps via absolute difference and projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Input channels: T1 + T2 + |T2 - T1| = 3 * in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, t1_feat: torch.Tensor, t2_feat: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(t2_feat - t1_feat)
        fused = torch.cat([t1_feat, t2_feat, abs_diff], dim=1)
        return self.conv(fused)


class SiameseChangeDetector(nn.Module):
    """Siamese Twin Network for Bitemporal Deforestation Detection."""

    def __init__(
        self,
        in_channels: int = 4,
        features: Optional[List[int]] = None,
        use_vit_bottleneck: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels per time step (e.g. 4 for RGB+NIR).
            features: Pyramid depths [stage1, stage2, stage3, stage4].
            use_vit_bottleneck: Whether to include ViT attention at bottleneck.
        """
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.features = features
        # Backbone encoder (shared weights between T1 and T2)
        self.encoder = UNet(
            in_channels=in_channels,
            num_classes=1,
            features=features,
            use_vit_bottleneck=use_vit_bottleneck,
        )

        # Multi-scale difference fusion blocks
        self.fuse1 = DifferenceFusionBlock(features[0], features[0])
        self.fuse2 = DifferenceFusionBlock(features[1], features[1])
        self.fuse3 = DifferenceFusionBlock(features[2], features[2])
        self.fuse4 = DifferenceFusionBlock(features[3], features[3])

        # Change decoder
        self.dec3 = DecoderBlock(features[3], features[2], features[2])
        self.dec2 = DecoderBlock(features[2], features[1], features[1])
        self.dec1 = DecoderBlock(features[1], features[0], features[0])

        # Final binary change prediction head
        self.head = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, features[0] // 2), num_channels=features[0] // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(features[0] // 2, 1, kernel_size=1),
        )

    def forward(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            t1: Pre-disturbance image (B, C, H, W).
            t2: Post-disturbance image (B, C, H, W).

        Returns:
            Change logits tensor of shape (B, 1, H, W).
            Values > 0 (or sigmoid > 0.5) denote deforestation.
        """
        feats_t1 = self.encoder.extract_features(t1)
        feats_t2 = self.encoder.extract_features(t2)

        f1 = self.fuse1(feats_t1[0], feats_t2[0])
        f2 = self.fuse2(feats_t1[1], feats_t2[1])
        f3 = self.fuse3(feats_t1[2], feats_t2[2])
        f4 = self.fuse4(feats_t1[3], feats_t2[3])

        d3 = self.dec3(f4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        change_logits = self.head(d1)
        return change_logits
