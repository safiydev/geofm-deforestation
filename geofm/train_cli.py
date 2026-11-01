"""Command-line training script for Geospatial AI Deforestation Detection.

Supports standard training, LoRA PEFT parameter-efficient fine-tuning,
spatial block cross-validation, and compound focal-tversky loss functions.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path
from geofm.models.siamese import SiameseChangeDetector
from geofm.models.cross_attention import MultimodalFusionDeforestationNet
from geofm.models.unet import UNet
from geofm.models.lora import apply_lora_to_model
from geofm.losses.focal_tversky import CompoundDeforestationLoss, FocalTverskyLoss
from geofm.losses.focal import BinaryFocalLoss
from geofm.training.dataset import (
    GeospatialDeforestationDataset,
    LazyGeospatialDataset,
    SpatialBlockSplitter,
)
from geofm.training.trainer import DeforestationTrainer
from geofm.utils.synthetic_data import (
    generate_synthetic_dataset,
    stream_and_cache_synthetic_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoFM Deforestation Detection Network")
    parser.add_argument("--model-type", type=str, default="siamese", choices=["siamese", "multimodal", "unet"])
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA parameter-efficient fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling factor")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (e.g. 8, 16, 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="compound", choices=["compound", "focal_tversky", "focal"])
    parser.add_argument("--num-samples", type=int, default=16, help="Synthetic dataset size for training")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing cached .npz chips")
    parser.add_argument("--cache-dir", type=str, default=None, help="Alias for --data-dir")
    parser.add_argument("--force-generate", action="store_true", help="Force regenerate synthetic data")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches to prefetch per worker")
    parser.add_argument("--ram-cache-gb", type=float, default=None, help="Host RAM budget for chip caching in GB")
    parser.add_argument("--no-preload", action="store_true", help="Disable preloading chips into RAM cache")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (FP16)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp", help="Disable Automatic Mixed Precision")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  GeoFM High-Throughput Deforestation Training Pipeline")
    print("=" * 65)
    print(f"Model Architecture: {args.model_type.upper()}")
    print(f"PEFT / LoRA Active: {args.use_lora} (Rank: {args.lora_rank})")
    print(f"Loss Function:      {args.loss}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Hardware Compute:   AMP (FP16) = {args.use_amp} | Workers = {args.num_workers}")
    print("-" * 65)

    # 1. Dataset Loading (High-Speed Memory Budgeted Caching)
    cache_dir = args.data_dir or args.cache_dir or "./data/chips_cache"
    manifest_file = os.path.join(cache_dir, "manifest.jsonl")

    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"System RAM: {vm.used / 1e9:.2f} GB used / {vm.total / 1e9:.2f} GB total ({vm.available / 1e9:.2f} GB available)")
    except Exception:
        pass

    has_cache = os.path.exists(manifest_file) or (os.path.isdir(cache_dir) and any(Path(cache_dir).glob("*.npz")))
    if has_cache and not args.force_generate:
        print(f"1. Loading chip cache from {cache_dir} with bounded RAM acceleration...")
        dataset = LazyGeospatialDataset(
            cache_dir=cache_dir,
            max_ram_cache_gb=args.ram_cache_gb,
            preload_cache=not args.no_preload,
        )
    else:
        print(f"1. Generating {args.num_samples} chips via zero-RAM disk streaming...")
        stream_and_cache_synthetic_dataset(
            output_dir=cache_dir,
            num_samples=args.num_samples,
            height=64,
            width=64,
            show_progress=True,
        )
        dataset = LazyGeospatialDataset(
            cache_dir=cache_dir,
            max_ram_cache_gb=args.ram_cache_gb,
            preload_cache=not args.no_preload,
        )

    print(f"   Dataset size: {len(dataset)} chips indexed.")
    splitter = SpatialBlockSplitter(grid_size_deg=0.1, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0)
    train_idx, val_idx, _ = splitter.split(dataset)

    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_sub,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )

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
        model, lora_stats = apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)
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

    # 5. Trainer with Hardware Maximization
    trainer = DeforestationTrainer(
        model=model,
        criterion=criterion,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    # 6. Run Training Loop
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    print("=" * 65)
    print(f"Training completed successfully! Best Validation IoU: {trainer.best_val_iou * 100:.2f}%")
    print(f"Checkpoints saved to: {os.path.abspath(args.checkpoint_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
