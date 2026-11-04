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
        learning_rate: float = 1e-3,
        gradient_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        model_type: str = "siamese",  # 'siamese', 'multimodal', or 'unet'
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        enable_tensorboard: bool = True,
        epoch_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Optimize cuDNN kernels for fixed image sizes on NVIDIA GPUs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)
        self.model_type = model_type
        self.learning_rate = float(learning_rate)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

        # Automatic Mixed Precision (AMP) on CUDA
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None

        self.criterion = (
            criterion if criterion is not None else CompoundDeforestationLoss()
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.learning_rate,
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
        """Route batch tensors according to model type with non_blocking asynchronous GPU transfers."""
        targets = batch["target"].to(self.device, non_blocking=True)

        if self.model_type == "siamese":
            t1 = batch["t1"].to(self.device, non_blocking=True)
            t2 = batch["t2"].to(self.device, non_blocking=True)
            logits = self.model(t1, t2)
        elif self.model_type == "multimodal":
            opt = batch["t2"].to(self.device, non_blocking=True)
            sar = batch["sar"].to(self.device, non_blocking=True)
            cloud_mask = batch.get("cloud_mask")
            if cloud_mask is not None:
                cloud_mask = cloud_mask.to(self.device, non_blocking=True)
            logits = self.model(opt, sar, cloud_mask=cloud_mask)
        else:
            # Standard single image U-Net
            x = batch["t2"].to(self.device, non_blocking=True)
            logits = self.model(x)

        return logits, targets

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch with AMP acceleration and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits, targets = self._forward_step(batch)
                    loss = self.criterion(logits, targets)
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.gradient_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                logits, targets = self._forward_step(batch)
                loss = self.criterion(logits, targets)
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.gradient_clip_norm,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item() * self.gradient_accumulation_steps)
            n_batches += 1

        return total_loss / max(1, n_batches)

    def validate_epoch(
        self, dataloader: DataLoader, threshold: float = 0.5
    ) -> Tuple[float, MetricReport]:
        """Run validation epoch and evaluate change detection metrics with AMP."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        metrics = DeforestationMetrics(threshold=threshold)

        with torch.no_grad():
            for batch in dataloader:
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits, targets = self._forward_step(batch)
                        loss = self.criterion(logits, targets)
                else:
                    logits, targets = self._forward_step(batch)
                    loss = self.criterion(logits, targets)

                total_loss += float(loss.item())
                n_batches += 1

                metrics.update(logits, targets)

        val_loss = total_loss / max(1, n_batches)
        report = metrics.compute()
        return val_loss, report

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        val_metrics: Optional[Union[MetricReport, Dict[str, Any]]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a comprehensive training checkpoint containing model, optimizer, scheduler, and metrics."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        metrics_dict = {}
        if val_metrics is not None:
            metrics_dict = val_metrics.to_dict() if hasattr(val_metrics, "to_dict") else dict(val_metrics)
        elif self.history:
            metrics_dict = self.history[-1]

        state = {
            "epoch": epoch,
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "best_val_iou": float(self.best_val_iou),
            "best_epoch": int(self.best_epoch),
            "patience_counter": int(self.patience_counter),
            "metrics": metrics_dict,
            "history": self.history,
            "learning_rate": self.optimizer.param_groups[0]["lr"] if self.optimizer else self.learning_rate,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            state.update(extra_meta)

        torch.save(state, filepath)
        return filepath

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint weights, optimizer, scheduler, scaler, and training history.

        Args:
            checkpoint_path: Path to .pt checkpoint file.

        Returns:
            int: The epoch index stored in the checkpoint (resumption starts at epoch + 1).
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 1. Load model weights
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"⚠️ Warning during model state dict loading ({e}), attempting with strict=False...")
            self.model.load_state_dict(state_dict, strict=False)

        resumed_epoch = 0
        if isinstance(checkpoint, dict):
            # 2. Restore optimizer state if present
            if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e:
                    print(f"⚠️ Warning: Could not restore optimizer state ({e}). Using fresh optimizer.")

            # 3. Restore scheduler state if present
            if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    print(f"⚠️ Warning: Could not restore scheduler state ({e}).")

            # 4. Restore AMP GradScaler state if present
            if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception as e:
                    print(f"⚠️ Warning: Could not restore scaler state ({e}).")

            # 5. Restore tracking metrics & history
            self.best_val_iou = float(checkpoint.get("best_val_iou", -1.0))
            self.best_epoch = int(checkpoint.get("best_epoch", 0))
            self.patience_counter = int(checkpoint.get("patience_counter", 0))

            loaded_history = checkpoint.get("history", [])
            if isinstance(loaded_history, list) and loaded_history:
                self.history = list(loaded_history)

            resumed_epoch = int(checkpoint.get("epoch", len(self.history)))

            # If best_val_iou was not saved explicitly, reconstruct from history
            if self.best_val_iou < 0 and self.history:
                for rec in self.history:
                    iou = rec.get("val_iou", -1.0)
                    if iou > self.best_val_iou:
                        self.best_val_iou = iou
                        self.best_epoch = rec.get("epoch", 0)

        best_iou_pct = f"{self.best_val_iou * 100:.2f}%" if self.best_val_iou >= 0 else "N/A"
        print(
            f"✓ Loaded checkpoint: {checkpoint_path}\n"
            f"  • Resumed Epoch:    {resumed_epoch}\n"
            f"  • Best Val IoU:     {best_iou_pct} (Epoch {self.best_epoch})\n"
            f"  • History Entries:  {len(self.history)}\n"
            f"  • Patience Counter: {self.patience_counter}/{self.early_stopping_patience}"
        )
        return resumed_epoch

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        eval_threshold: float = 0.5,
        start_epoch: int = 1,
    ) -> List[Dict[str, Any]]:
        """Execute full training and validation lifecycle."""
        if start_epoch > epochs and epochs > 0:
            print(f"Training already completed up to epoch {start_epoch - 1} (requested epochs: {epochs}).")
            return self.history

        if self.device.type == "cuda":
            device_name = torch.cuda.get_device_name(self.device)
            vram_total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            print(f"Starting training on {device_name} ({vram_total_gb:.1f} GB VRAM) from epoch {start_epoch} to {epochs}.")
            print(f"  ⚡ GPU Acceleration: AMP (FP16) = {self.use_amp} | cuDNN Benchmark = True | Grad Accum = {self.gradient_accumulation_steps}")
        else:
            print(f"Starting training on device: {self.device} from epoch {start_epoch} to {epochs}.")
        epoch = start_epoch - 1

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader, threshold=eval_threshold)

            prev_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr < prev_lr:
                print(f"  📉 [ReduceLROnPlateau] Validation loss plateau detected. Learning rate reduced: {prev_lr:.2e} -> {current_lr:.2e}")

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
            if self.device.type == "cuda":
                epoch_record["gpu_vram_used_gb"] = round(torch.cuda.max_memory_allocated(self.device) / (1024**3), 2)

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
                self.save_checkpoint(best_path, epoch=epoch, val_metrics=val_metrics)
            else:
                self.patience_counter += 1

            # Save latest checkpoint at the end of every epoch for instant crash recovery
            latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            self.save_checkpoint(latest_path, epoch=epoch, val_metrics=val_metrics)

            # Also maintain last_model.pt for backwards compatibility
            last_path = os.path.join(self.checkpoint_dir, "last_model.pt")
            self.save_checkpoint(last_path, epoch=epoch, val_metrics=val_metrics)

            # Export metrics JSON & CSV dynamically after each epoch
            self.export_metrics_json()
            self.export_metrics_csv()

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        # Save last checkpoint if not saved yet (e.g. 0 epochs)
        last_path = os.path.join(self.checkpoint_dir, "last_model.pt")
        if not os.path.exists(last_path):
            self.save_checkpoint(last_path, epoch=epoch)

        # Final export of metrics JSON & CSV
        self.export_metrics_json()
        self.export_metrics_csv()

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

        # Maintain latest_metrics.json and metrics_history.json in checkpoint_dir
        for name in ("latest_metrics.json", "metrics_history.json"):
            target_json = os.path.join(self.checkpoint_dir, name)
            try:
                with open(target_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except OSError:
                pass

        return out_path

    def export_metrics_csv(self, filepath: Optional[str] = None) -> str:
        """Export epoch history to a CSV file for file-based tracking."""
        out_path = filepath or os.path.join(self.log_dir, f"{self.experiment_name}_metrics.csv")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_f1",
            "val_recall",
            "val_precision",
            "val_specificity",
            "val_accuracy",
            "lr",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        lines = [",".join(header)]

        for h in self.history:
            cm = h.get("confusion_matrix", {})
            row = [
                str(h.get("epoch", "")),
                str(h.get("train_loss", "")),
                str(h.get("val_loss", "")),
                str(h.get("val_iou", "")),
                str(h.get("val_f1", "")),
                str(h.get("val_recall", "")),
                str(h.get("val_precision", "")),
                str(h.get("val_specificity", "")),
                str(h.get("val_accuracy", "")),
                str(h.get("lr", "")),
                str(cm.get("tp", 0)),
                str(cm.get("fp", 0)),
                str(cm.get("fn", 0)),
                str(cm.get("tn", 0)),
            ]
            lines.append(",".join(row))

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        for name in ("latest_metrics.csv", "metrics_history.csv"):
            target_csv = os.path.join(self.checkpoint_dir, name)
            try:
                with open(target_csv, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
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
        """Flush and return TensorBoard event log directory if active, or file log_dir."""
        if self.writer is not None:
            self.writer.flush()
            return getattr(self.writer, "log_dir", self.log_dir)
        return self.log_dir

    def close(self) -> None:
        """Safely close TensorBoard writer and flush buffers."""
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None
