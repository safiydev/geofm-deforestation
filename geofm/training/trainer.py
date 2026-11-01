"""TerraTorch / PyTorch Lightning style Trainer for geospatial deforestation models.

Handles training, validation, gradient clipping, CosineAnnealing learning rate scheduling,
model checkpointing, early stopping, and metric tracking.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from geofm.losses.focal_tversky import CompoundDeforestationLoss
from geofm.training.metrics import DeforestationMetrics, MetricReport

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


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
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        enable_tensorboard: bool = True,
        epoch_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
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

        self.log_dir = log_dir or os.path.join(checkpoint_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiment_name = experiment_name or f"run_{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.enable_tensorboard = enable_tensorboard
        self.epoch_callback = epoch_callback

        self.writer = None
        if self.enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_exp_dir = os.path.join(self.log_dir, self.experiment_name)
            self.writer = SummaryWriter(log_dir=tb_exp_dir)

        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience

        self.history: List[Dict[str, Any]] = []
        self.best_val_iou: float = -1.0
        self.best_epoch: int = 0
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
    ) -> List[Dict[str, Any]]:
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
                "val_specificity": round(val_metrics.specificity, 4),
                "val_accuracy": round(val_metrics.overall_accuracy, 4),
                "lr": current_lr,
                "confusion_matrix": {
                    "tp": int(val_metrics.true_positives),
                    "fp": int(val_metrics.false_positives),
                    "fn": int(val_metrics.false_negatives),
                    "tn": int(val_metrics.true_negatives),
                },
            }
            self.history.append(epoch_record)

            # TensorBoard logging
            if self.writer is not None:
                try:
                    self.writer.add_scalar("Loss/train", train_loss, epoch)
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                    self.writer.add_scalar("Metrics/IoU", val_metrics.iou, epoch)
                    self.writer.add_scalar("Metrics/F1", val_metrics.f1_score, epoch)
                    self.writer.add_scalar("Metrics/Precision", val_metrics.precision, epoch)
                    self.writer.add_scalar("Metrics/Recall", val_metrics.recall, epoch)
                    self.writer.add_scalar("Metrics/Specificity", val_metrics.specificity, epoch)
                    self.writer.add_scalar("Metrics/Accuracy", val_metrics.overall_accuracy, epoch)
                    self.writer.add_scalar("LearningRate", current_lr, epoch)
                    self.writer.add_scalar("ConfusionMatrix/TP", val_metrics.true_positives, epoch)
                    self.writer.add_scalar("ConfusionMatrix/FP", val_metrics.false_positives, epoch)
                    self.writer.add_scalar("ConfusionMatrix/FN", val_metrics.false_negatives, epoch)
                    self.writer.add_scalar("ConfusionMatrix/TN", val_metrics.true_negatives, epoch)
                    self.writer.flush()
                except Exception as tb_err:
                    print(f"Warning: Failed to log scalars to TensorBoard: {tb_err}")

            # Optional callback for live UI updates
            if self.epoch_callback is not None:
                try:
                    self.epoch_callback(epoch, epoch_record)
                except Exception as cb_err:
                    print(f"Warning: Epoch callback error: {cb_err}")

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
                self.best_epoch = epoch
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

        # Export metrics JSON
        self.export_metrics_json()

        return self.history

    def export_metrics_json(self, filepath: Optional[str] = None) -> str:
        """Export epoch history and metrics summary to a clean JSON file."""
        out_path = filepath or os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        summary = self.get_metrics_summary()
        payload = {
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "best_epoch": self.best_epoch,
            "best_val_iou": round(self.best_val_iou, 4),
            "total_epochs_recorded": len(self.history),
            "summary": summary,
            "history": self.history,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Also maintain a latest_metrics.json in checkpoint_dir
        latest_path = os.path.join(self.checkpoint_dir, "latest_metrics.json")
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError:
            pass

        return out_path

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Compute high-level summary statistics from training history."""
        if not self.history:
            return {
                "status": "not_started",
                "best_val_iou": 0.0,
                "best_epoch": 0,
                "epochs_completed": 0,
            }

        train_losses = [h["train_loss"] for h in self.history]
        val_losses = [h["val_loss"] for h in self.history]
        val_ious = [h["val_iou"] for h in self.history]
        val_f1s = [h["val_f1"] for h in self.history]

        latest = self.history[-1]
        return {
            "status": "completed",
            "epochs_completed": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_iou": round(self.best_val_iou, 4),
            "min_train_loss": round(min(train_losses), 4),
            "min_val_loss": round(min(val_losses), 4),
            "latest_train_loss": latest["train_loss"],
            "latest_val_loss": latest["val_loss"],
            "latest_val_iou": latest["val_iou"],
            "latest_val_f1": latest["val_f1"],
            "latest_val_recall": latest["val_recall"],
            "latest_val_precision": latest["val_precision"],
            "latest_confusion_matrix": latest.get("confusion_matrix", {}),
            "loss_curve": {
                "train": train_losses,
                "val": val_losses,
            },
            "iou_curve": val_ious,
            "f1_curve": val_f1s,
        }

    def export_tensorboard(self, output_dir: Optional[str] = None) -> Optional[str]:
        """Flush and return TensorBoard event log directory if active."""
        if self.writer is not None:
            self.writer.flush()
            return getattr(self.writer, "log_dir", self.log_dir)
        return None

    def close(self) -> None:
        """Safely close TensorBoard writer and flush buffers."""
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None
