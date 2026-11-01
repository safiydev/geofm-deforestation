"""Command-line training script for Geospatial AI Deforestation Detection.

Supports standard training, LoRA PEFT parameter-efficient fine-tuning,
spatial block cross-validation, and compound focal-tversky loss functions.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

from geofm.models.siamese import SiameseChangeDetector
from geofm.models.cross_attention import MultimodalFusionDeforestationNet
from geofm.models.unet import UNet
from geofm.models.lora import apply_lora_to_model
from geofm.losses.focal_tversky import CompoundDeforestationLoss, FocalTverskyLoss
from geofm.losses.focal import BinaryFocalLoss
from geofm.training.dataset import (
    GeospatialDeforestationDataset,
    SpatialBlockSplitter,
)
from geofm.training.trainer import DeforestationTrainer
from geofm.utils.synthetic_data import generate_synthetic_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoFM Deforestation Detection Network")
    parser.add_argument("--model-type", type=str, default="siamese", choices=["siamese", "multimodal", "unet"])
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA parameter-efficient fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="compound", choices=["compound", "focal_tversky", "focal"])
    parser.add_argument("--num-samples", type=int, default=16, help="Synthetic dataset size for training")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  GeoFM Deforestation Model Training Pipeline")
    print("=" * 65)
    print(f"Model Architecture: {args.model_type.upper()}")
    print(f"PEFT / LoRA Active: {args.use_lora} (Rank: {args.lora_rank})")
    print(f"Loss Function:      {args.loss}")
    print(f"Epochs:             {args.epochs}")
    print(f"Learning Rate:      {args.lr}")
    print("-" * 65)

    # 1. Dataset Generation & Spatial Block Cross-Validation Split
    print("1. Generating geospatial tiles with geographic coordinates...")
    samples = generate_synthetic_dataset(num_samples=args.num_samples, height=64, width=64)
    splitter = SpatialBlockSplitter(grid_size_deg=0.1, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0)
    train_idx, val_idx, _ = splitter.split(samples)

    dataset = GeospatialDeforestationDataset(samples)
    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)

    print(f"   Spatial split complete: {len(train_sub)} train samples, {len(val_sub)} val samples.")

    # 2. Model Initialization
    print("2. Initializing model architecture...")
    if args.model_type == "siamese":
        model = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
    elif args.model_type == "multimodal":
        model = MultimodalFusionDeforestationNet(opt_channels=4, sar_channels=3, features=[16, 32, 64, 128])
    else:
        model = UNet(in_channels=4, num_classes=1, features=[16, 32, 64, 128])

    # 3. LoRA PEFT injection if requested
    if args.use_lora:
        print("3. Injecting LoRA low-rank decomposition adapters into attention layers...")
        model, lora_stats = apply_lora_to_model(model, rank=args.lora_rank, alpha=16.0)
        print(f"   Total Parameters:     {lora_stats['total_parameters']:,}")
        print(f"   Trainable Parameters: {lora_stats['trainable_parameters']:,} ({lora_stats['trainable_fraction']*100:.2f}%)")
        print(f"   Adapted Layers:       {lora_stats['adapted_layers']}")

    # 4. Loss Function
    if args.loss == "compound":
        criterion = CompoundDeforestationLoss(alpha=0.75, beta=0.25)
    elif args.loss == "focal_tversky":
        criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
    else:
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)

    # 5. Trainer
    trainer = DeforestationTrainer(
        model=model,
        criterion=criterion,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
    )

    # 6. Run Training Loop
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    print("=" * 65)
    print(f"Training completed successfully! Best Validation IoU: {trainer.best_val_iou * 100:.2f}%")
    print(f"Checkpoints saved to: {os.path.abspath(args.checkpoint_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
