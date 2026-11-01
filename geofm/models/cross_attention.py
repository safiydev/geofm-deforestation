"""Hybrid Cross-Attention Fusion (Prithvi-CAFE style) for Multimodal Optical + SAR.

Performs bidirectional cross-attention between Sentinel-2/NICFI optical tokens
and Sentinel-1 SAR backscatter tokens, with adaptive cloud-gating to fall back
gracefully to SAR structural backscatter when optical imagery is cloud-occluded.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from geofm.models.unet import UNet, DecoderBlock, ViTMultiHeadAttention


class CrossAttentionBlock(nn.Module):
    """Bidirectional Cross-Attention between Optical and SAR feature maps."""

    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Optical queries SAR (extracts structural context to complement optical)
        self.norm_opt1 = nn.LayerNorm(dim)
        self.norm_sar1 = nn.LayerNorm(dim)
        self.mha_opt_to_sar = ViTMultiHeadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout
        )

        # SAR queries Optical (extracts spectral context to complement radar)
        self.norm_opt2 = nn.LayerNorm(dim)
        self.norm_sar2 = nn.LayerNorm(dim)
        self.mha_sar_to_opt = ViTMultiHeadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout
        )

        # Feed-forward refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm_fused = nn.LayerNorm(dim)

        # Cloud gating linear projection
        self.gate_proj = nn.Linear(1, 1)

    def forward(
        self,
        feat_opt: torch.Tensor,
        feat_sar: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = feat_opt.shape
        opt_seq = feat_opt.flatten(2).transpose(1, 2)
        sar_seq = feat_sar.flatten(2).transpose(1, 2)

        # 1. Optical attends to SAR
        q_opt = self.norm_opt1(opt_seq)
        k_sar = self.norm_sar1(sar_seq)
        attn_opt = self.mha_opt_to_sar(query=q_opt, key=k_sar, value=sar_seq)
        opt_enhanced = opt_seq + attn_opt

        # 2. SAR attends to Optical
        q_sar = self.norm_sar2(sar_seq)
        k_opt = self.norm_opt2(opt_seq)
        attn_sar = self.mha_sar_to_opt(query=q_sar, key=k_opt, value=opt_seq)
        sar_enhanced = sar_seq + attn_sar

        # 3. Adaptive Cloud Gating
        if cloud_mask is not None:
            cm = F.interpolate(cloud_mask.float(), size=(h, w), mode="nearest")
            cm_seq = cm.flatten(2).transpose(1, 2)  # (B, N, 1)
            gate = torch.sigmoid(self.gate_proj(1.0 - cm_seq))
            opt_enhanced = opt_enhanced * gate

        # 4. Concatenate and project back
        cat_tokens = torch.cat([opt_enhanced, sar_enhanced], dim=-1)
        fused_tokens = self.mlp(cat_tokens) + opt_enhanced
        fused_tokens = self.norm_fused(fused_tokens)

        return fused_tokens.transpose(1, 2).reshape(b, c, h, w)


class MultimodalFusionDeforestationNet(nn.Module):
    """End-to-End Multimodal Optical + SAR Deforestation Detection Network."""

    def __init__(
        self,
        opt_channels: int = 4,
        sar_channels: int = 3,
        features: Optional[list] = None,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.opt_encoder = UNet(in_channels=opt_channels, num_classes=1, features=features)
        self.sar_encoder = UNet(in_channels=sar_channels, num_classes=1, features=features)

        self.fusion = CrossAttentionBlock(dim=features[3], num_heads=4)

        self.dec3 = DecoderBlock(features[3], features[2], features[2])
        self.dec2 = DecoderBlock(features[2], features[1], features[1])
        self.dec1 = DecoderBlock(features[1], features[0], features[0])

        self.head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(
        self,
        opt: torch.Tensor,
        sar: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        opt_feats = self.opt_encoder.extract_features(opt)
        sar_feats = self.sar_encoder.extract_features(sar)

        fused_bottleneck = self.fusion(opt_feats[3], sar_feats[3], cloud_mask=cloud_mask)

        skip3 = (opt_feats[2] + sar_feats[2]) * 0.5
        skip2 = (opt_feats[1] + sar_feats[1]) * 0.5
        skip1 = (opt_feats[0] + sar_feats[0]) * 0.5

        d3 = self.dec3(fused_bottleneck, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.head(d1)
