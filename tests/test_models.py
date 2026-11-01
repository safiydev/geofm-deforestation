"""Tests for model architectures: U-Net, Siamese Change Detector, Cross-Attention Fusion, and LoRA."""

import torch
import pytest

from geofm.models.unet import UNet, ResBlock, ViTBottleneck
from geofm.models.siamese import SiameseChangeDetector
from geofm.models.cross_attention import MultimodalFusionDeforestationNet, CrossAttentionBlock
from geofm.models.lora import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_state_dict,
    load_lora_state_dict,
)


def test_unet_forward_and_shapes():
    model = UNet(in_channels=4, num_classes=1, features=[16, 32, 64, 128], use_vit_bottleneck=True)
    x = torch.randn(2, 4, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_siamese_change_detector():
    model = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
    t1 = torch.randn(2, 4, 64, 64)
    t2 = torch.randn(2, 4, 64, 64)
    logits = model(t1, t2)
    assert logits.shape == (2, 1, 64, 64)


def test_cross_attention_multimodal_fusion():
    model = MultimodalFusionDeforestationNet(
        opt_channels=4, sar_channels=3, features=[16, 32, 64, 128]
    )
    opt = torch.randn(2, 4, 64, 64)
    sar = torch.randn(2, 3, 64, 64)
    cloud_mask = torch.zeros(2, 1, 64, 64)
    cloud_mask[:, :, 10:25, 10:25] = 1.0  # Simulated cloud patch

    out = model(opt, sar, cloud_mask=cloud_mask)
    assert out.shape == (2, 1, 64, 64)


def test_lora_adaptation_and_merge():
    model = UNet(in_channels=4, num_classes=1, features=[16, 32, 64, 128])
    base_params = sum(p.numel() for p in model.parameters())

    adapted_model, stats = apply_lora_to_model(model, rank=4, alpha=8.0, target_names=["q_proj", "v_proj"])
    assert stats["adapted_layers"] >= 2
    assert stats["trainable_fraction"] < 0.05  # Over 95% parameters frozen

    # Test forward pass with LoRA
    x = torch.randn(1, 4, 32, 32)
    out1 = adapted_model(x)
    assert out1.shape == (1, 1, 32, 32)

    # Test state dict extraction
    lora_sd = get_lora_state_dict(adapted_model)
    assert len(lora_sd) > 0
    assert all("lora_" in k for k in lora_sd.keys())

    # Test merge and unmerge
    for m in adapted_model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
            assert m.merged == True
            m.unmerge()
            assert m.merged == False
