"""TerraTorch / PyTorch Lightning style Trainer for geospatial deforestation models.

Handles training, validation, gradient clipping, CosineAnnealing learning rate scheduling,
model checkpointing, early stopping, and metric tracking.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from geofm.losses.focal_tversky import CompoundDeforestationLoss
from geofm.training.metrics import DeforestationMetrics, MetricReport


class DeforestationTrainer:
    """Trainer for training and fine-tuning deforestation detection networks."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[any] = None,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: str = "./checkpoints",
        gradient_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        model_type: str = "siamese",  # 'siamese', 'multimodal', or 'unet'
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model_type = model_type
        self.criterion = (
            criterion if criterion is not None else CompoundDeforestationLoss()
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=1e-3,
                weight_decay=1e-4,
            )
        )
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience

        self.history: List[Dict[str, float]] = []
        self.best_val_iou: float = -1.0
        self.patience_counter: int = 0

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route batch tensors according to model type."""
        targets = batch["target"].to(self.device)

        if self.model_type == "siamese":
            t1 = batch["t1"].to(self.device)
            t2 = batch["t2"].to(self.device)
            logits = self.model(t1, t2)
        elif self.model_type == "multimodal":
            opt = batch["t2"].to(self.device)
            sar = batch["sar"].to(self.device)
            cloud_mask = batch.get("cloud_mask")
            if cloud_mask is not None:
                cloud_mask = cloud_mask.to(self.device)
            logits = self.model(opt, sar, cloud_mask=cloud_mask)
        else:
            # Standard single image U-Net
            x = batch["t2"].to(self.device)
            logits = self.model(x)

        return logits, targets

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            logits, targets = self._forward_step(batch)
            loss = self.criterion(logits, targets)

            loss.backward()
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip_norm,
                )
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(1, n_batches)

    def validate_epoch(
        self, dataloader: DataLoader, threshold: float = 0.5
    ) -> Tuple[float, MetricReport]:
        """Run validation epoch and evaluate change detection metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        metrics = DeforestationMetrics(threshold=threshold)

        with torch.no_grad():
            for batch in dataloader:
                logits, targets = self._forward_step(batch)
                loss = self.criterion(logits, targets)

                total_loss += float(loss.item())
                n_batches += 1

                metrics.update(logits, targets)

        val_loss = total_loss / max(1, n_batches)
        report = metrics.compute()
        return val_loss, report

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        eval_threshold: float = 0.5,
    ) -> List[Dict[str, float]]:
        """Execute full training and validation lifecycle."""
        print(f"Starting training on device: {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader, threshold=eval_threshold)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_iou": round(val_metrics.iou, 4),
                "val_f1": round(val_metrics.f1_score, 4),
                "val_recall": round(val_metrics.recall, 4),
                "val_precision": round(val_metrics.precision, 4),
                "lr": current_lr,
            }
            self.history.append(epoch_record)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val IoU: {val_metrics.iou * 100:.2f}% | "
                f"Val F1: {val_metrics.f1_score * 100:.2f}% | "
                f"Val Recall: {val_metrics.recall * 100:.2f}%"
            )

            # Checkpoint best model
            if val_metrics.iou > self.best_val_iou:
                self.best_val_iou = val_metrics.iou
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "metrics": val_metrics.to_dict(),
                        "history": self.history,
                    },
                    best_path,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, "last_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
            },
            last_path,
        )

        return self.history
