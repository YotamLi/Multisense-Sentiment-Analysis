"""Multi-task BERT trainer with safe AMP and reproducible checkpoints."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..evaluation.metrics import compute_metrics
from .checkpoint import load_checkpoint
from .multi_head_bert import MultiHeadBERT


class MultiTaskTrainer:
    """Train a shared BERT encoder with four task-specific heads."""

    def __init__(
        self,
        model: MultiHeadBERT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device | None = None,
        output_dir: str = "checkpoints",
        metadata: dict | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.metadata = metadata or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

        self.grad_accum_steps = max(
            1,
            int(config.get("gradient_accumulation_steps", 1)),
        )
        self.use_amp = bool(
            config.get("fp16", False) and self.device.type == "cuda"
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.best_score = float("-inf")
        self.best_val_metrics: dict = {}
        self.patience_counter = 0
        self.patience = int(config.get("early_stopping_patience", 3))
        self.min_delta = float(config.get("early_stopping_min_delta", 0.0))

        logger.info(
            f"Mixed precision: {'enabled' if self.use_amp else 'disabled'}"
        )
        logger.info(
            f"Gradient accumulation steps: {self.grad_accum_steps}"
        )

    def _build_optimizer(self) -> AdamW:
        lr_encoder = float(self.config.get("learning_rate_encoder", 2e-5))
        lr_heads = float(self.config.get("learning_rate_heads", 1e-3))
        weight_decay = float(self.config.get("weight_decay", 0.01))

        return AdamW(
            [
                {
                    "params": list(self.model.get_encoder_params()),
                    "lr": lr_encoder,
                    "weight_decay": weight_decay,
                },
                {
                    "params": list(self.model.get_head_params()),
                    "lr": lr_heads,
                    "weight_decay": 0.0,
                },
            ]
        )

    def _build_scheduler(self):
        updates_per_epoch = math.ceil(
            len(self.train_loader) / self.grad_accum_steps
        )
        epochs = int(self.config.get("epochs", 10))
        total_steps = max(1, updates_per_epoch * epochs)
        warmup_steps = int(
            total_steps * float(self.config.get("warmup_ratio", 0.1))
        )

        logger.info(
            "Scheduler steps: "
            f"{updates_per_epoch} updates/epoch, "
            f"{total_steps} total, {warmup_steps} warmup"
        )

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _compute_monitor_score(
        self,
        val_metrics: dict[str, dict],
    ) -> tuple[float, dict[str, float]]:
        metric_name = self.config.get("monitor_metric", "macro_f1")
        configured_weights = self.config.get("monitor_weights", {})

        task_scores: dict[str, float] = {}
        for task_name in MultiHeadBERT.TASK_NAMES:
            metrics = val_metrics.get(task_name)
            if isinstance(metrics, dict) and metrics.get(metric_name) is not None:
                task_scores[task_name] = float(metrics[metric_name])

        if not task_scores:
            logger.warning("No validation metrics were available for monitoring")
            return 0.0, {}

        raw_weights: dict[str, float] = {}
        for task_name in task_scores:
            try:
                raw_weights[task_name] = max(
                    0.0,
                    float(configured_weights.get(task_name, 1.0)),
                )
            except (TypeError, ValueError):
                raw_weights[task_name] = 1.0

        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            raw_weights = {task: 1.0 for task in task_scores}
            total_weight = float(len(raw_weights))

        normalised_weights = {
            task: weight / total_weight
            for task, weight in raw_weights.items()
        }
        score = sum(
            task_scores[task] * normalised_weights[task]
            for task in task_scores
        )

        summary = ", ".join(
            f"{task}={task_scores[task]:.4f}×{normalised_weights[task]:.2f}"
            for task in task_scores
        )
        logger.info(f"Validation monitor score: {score:.4f} ({summary})")
        return score, normalised_weights

    def train(self) -> dict:
        epochs = int(self.config.get("epochs", 10))
        history = {
            "train_loss": [],
            "val_metrics": [],
            "monitor_score": [],
        }
        logger.info(f"Starting training for {epochs} epochs on {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            monitor_score, monitor_weights = self._compute_monitor_score(
                val_metrics
            )

            history["train_loss"].append(train_loss)
            history["val_metrics"].append(val_metrics)
            history["monitor_score"].append(monitor_score)

            improved = monitor_score > self.best_score + self.min_delta
            if improved:
                previous_best = self.best_score
                self.best_score = monitor_score
                self.best_val_metrics = val_metrics
                self.patience_counter = 0
                self._save_checkpoint(
                    epoch=epoch,
                    monitor_score=monitor_score,
                    val_metrics=val_metrics,
                    monitor_weights=monitor_weights,
                )
                improvement = (
                    "first valid score"
                    if previous_best == float("-inf")
                    else f"+{monitor_score - previous_best:.4f}"
                )
                logger.info(
                    "New best model saved — "
                    f"overall validation score {monitor_score:.4f} "
                    f"({improvement})"
                )
            else:
                self.patience_counter += 1
                logger.info(
                    "Validation score did not improve: "
                    f"current={monitor_score:.4f}, best={self.best_score:.4f}, "
                    f"patience={self.patience_counter}/{self.patience}"
                )
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}"
                    )
                    break

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        valid_batches = 0
        optimiser_updates = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )
        total_batches = len(self.train_loader)
        accumulated = 0
        current_group_size = min(self.grad_accum_steps, total_batches)

        for step, batch in enumerate(progress):
            if accumulated == 0:
                current_group_size = min(
                    self.grad_accum_steps,
                    total_batches - step,
                )

            batch = {key: value.to(self.device) for key, value in batch.items()}
            amp_context = (
                autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )

            with amp_context:
                output = self.model(**batch)
                if output.loss is None:
                    logger.warning(f"Skipping batch {step}: no valid labels")
                    continue
                original_loss = output.loss
                scaled_loss = original_loss / current_group_size

            self.scaler.scale(scaled_loss).backward()
            accumulated += 1
            total_loss += float(original_loss.item())
            valid_batches += 1

            if accumulated >= current_group_size:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimiser_updates += 1
                accumulated = 0

            task_loss_display = {
                f"{task}_loss": f"{loss.item():.4f}"
                for task, loss in output.per_task_losses.items()
            }
            progress.set_postfix(
                loss=f"{original_loss.item():.4f}",
                updates=optimiser_updates,
                **task_loss_display,
            )

        # Defensive flush for any unusual skipped-batch pattern.
        if accumulated > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            optimiser_updates += 1

        average_loss = total_loss / max(1, valid_batches)
        logger.info(
            f"Epoch {epoch} — Train Loss: {average_loss:.4f}, "
            f"Optimizer Updates: {optimiser_updates}"
        )
        return average_loss

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, dict]:
        self.model.eval()
        task_preds = {task: [] for task in MultiHeadBERT.TASK_NAMES}
        task_labels = {task: [] for task in MultiHeadBERT.TASK_NAMES}

        for batch in tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            leave=False,
        ):
            batch_device = {
                key: value.to(self.device) for key, value in batch.items()
            }
            output = self.model(**batch_device)
            logits_map = {
                "sentiment": output.sentiment_logits,
                "emotion": output.emotion_logits,
                "intensity": output.intensity_logits,
                "topic": output.topic_logits,
            }

            for task_name, logits in logits_map.items():
                label_key = f"{task_name}_labels"
                labels = batch[label_key].cpu()
                predictions = logits.argmax(dim=-1).cpu()
                valid_mask = labels != -1
                if valid_mask.any():
                    task_preds[task_name].extend(
                        predictions[valid_mask].tolist()
                    )
                    task_labels[task_name].extend(labels[valid_mask].tolist())

        results: dict[str, dict] = {}
        for task_name in MultiHeadBERT.TASK_NAMES:
            if task_labels[task_name]:
                metrics = compute_metrics(
                    task_labels[task_name],
                    task_preds[task_name],
                )
                results[task_name] = metrics
                logger.info(
                    f"Epoch {epoch} — Val {task_name}: "
                    f"Acc={metrics['accuracy']:.4f}, "
                    f"F1={metrics['macro_f1']:.4f}"
                )

        return results

    def _save_checkpoint(
        self,
        epoch: int,
        monitor_score: float,
        val_metrics: dict,
        monitor_weights: dict[str, float],
    ) -> None:
        path = self.output_dir / "best_model.pt"
        sentiment_f1 = float(
            val_metrics.get("sentiment", {}).get("macro_f1", 0.0)
        )
        checkpoint = {
            "format_version": 2,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.use_amp else None
            ),
            "monitor_score": float(monitor_score),
            "monitor_metric": self.config.get("monitor_metric", "macro_f1"),
            "monitor_weights": monitor_weights,
            "val_metrics": val_metrics,
            "best_f1": sentiment_f1,
            "training_config": self.config,
            "metadata": self.metadata,
            "torch_version": torch.__version__,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint written to: {path}")

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        monitor_score = checkpoint.get(
            "monitor_score",
            checkpoint.get("best_f1", 0.0),
        )
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch={checkpoint.get('epoch', 'unknown')}, "
            f"monitor_score={float(monitor_score):.4f})"
        )
        return checkpoint
