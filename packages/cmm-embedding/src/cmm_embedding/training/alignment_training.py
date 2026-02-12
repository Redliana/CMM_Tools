from __future__ import annotations

# embedding/training/alignment_training.py
"""
Cross-Modal Alignment Training for CMM Embeddings

This module provides the training loop and utilities for training the
cross-modal alignment component of the CMM embedding architecture.

The training uses contrastive learning (CLIP-style) to align embeddings
from different modalities into a shared semantic space.

Usage:
    trainer = CrossModalAlignmentTrainer(config)
    trainer.train(train_dataloader, val_dataloader)
"""

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for cross-modal alignment training."""

    # Model settings
    embedding_dim: int = 768
    projection_dim: int = 512
    temperature: float = 0.07
    temperature_learnable: bool = True

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Loss settings
    use_confidence_weighting: bool = True
    use_hard_negatives: bool = False
    hard_negative_weight: float = 0.5

    # Optimization
    use_mixed_precision: bool = True
    scheduler_type: str = "cosine"  # "cosine" or "onecycle"

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    keep_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "cmm-embedding-alignment"
    wandb_run_name: str | None = None

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# Loss Functions
# =============================================================================


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent loss for contrastive learning.

    This is the standard contrastive loss used in CLIP and similar models.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        confidence_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute symmetric InfoNCE loss.

        Args:
            embeddings_a: Embeddings from modality A, shape (batch_size, dim)
            embeddings_b: Embeddings from modality B, shape (batch_size, dim)
            confidence_weights: Optional per-example weights, shape (batch_size,)

        Returns:
            tuple of (loss, metrics_dict)
        """
        batch_size = embeddings_a.size(0)
        device = embeddings_a.device

        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature.clamp(min=0.01)

        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)

        # Compute cross-entropy loss both ways
        loss_a_to_b = F.cross_entropy(logits, labels, reduction="none")
        loss_b_to_a = F.cross_entropy(logits.T, labels, reduction="none")

        loss = (loss_a_to_b + loss_b_to_a) / 2

        # Apply confidence weighting
        if confidence_weights is not None:
            loss = loss * confidence_weights

        loss = loss.mean()

        # Compute metrics
        with torch.no_grad():
            # Accuracy (how often the correct pair is ranked highest)
            preds_a = logits.argmax(dim=1)
            preds_b = logits.argmax(dim=0)
            acc_a = (preds_a == labels).float().mean()
            acc_b = (preds_b == labels).float().mean()

            # Mean positive/negative similarities
            pos_sim = logits.diag().mean()
            neg_sim = (logits.sum() - logits.diag().sum()) / (batch_size * (batch_size - 1))

        metrics = {
            "loss": loss.item(),
            "acc_a_to_b": acc_a.item(),
            "acc_b_to_a": acc_b.item(),
            "pos_similarity": pos_sim.item(),
            "neg_similarity": neg_sim.item(),
            "temperature": self.temperature.item()
            if isinstance(self.temperature, nn.Parameter)
            else self.temperature,
        }

        return loss, metrics


class HardNegativeLoss(nn.Module):
    """
    Contrastive loss with explicit hard negative mining.

    Hard negatives are examples that are similar but not paired.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        hard_negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with hard negatives.

        Args:
            anchor: Anchor embeddings, shape (batch_size, dim)
            positive: Positive embeddings, shape (batch_size, dim)
            hard_negatives: Hard negative embeddings, shape (batch_size, num_neg, dim)
        """
        batch_size = anchor.size(0)
        hard_negatives.size(1)

        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        hard_negatives = F.normalize(hard_negatives, p=2, dim=-1)

        # Positive similarities
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # (batch_size,)

        # Hard negative similarities
        neg_sim = (
            torch.bmm(hard_negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
        )  # (batch_size, num_neg)

        # Combine for softmax
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_neg)

        # Labels: positive is always index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)

        return loss


# =============================================================================
# Training Utilities
# =============================================================================


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, output_dir: str, keep_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self.checkpoints: list[tuple[float, str]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: dict[str, float],
    ) -> str:
        """Save a checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }

        path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)

        # Track checkpoint
        metric_value = metrics.get("val_loss", metrics.get("loss", 0))
        self.checkpoints.append((metric_value, str(path)))
        self.checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints
        while len(self.checkpoints) > self.keep_n:
            _, old_path = self.checkpoints.pop()
            if Path(old_path).exists():
                Path(old_path).unlink()

        logger.info(f"Saved checkpoint to {path}")
        return str(path)

    def load(self, path: str) -> dict[str, Any]:
        """Load a checkpoint."""
        return torch.load(path, map_location="cpu")

    def get_best_checkpoint(self) -> str | None:
        """Get path to best checkpoint."""
        if self.checkpoints:
            return self.checkpoints[0][1]
        return None


class MetricTracker:
    """Track training metrics."""

    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.step_metrics: dict[str, float] = {}

    def update(self, metrics: dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.step_metrics[key] = value

    def get_average(self, key: str, window: int = 100) -> float:
        """Get moving average of a metric."""
        values = self.metrics.get(key, [])
        if not values:
            return 0.0
        return np.mean(values[-window:])

    def get_all_averages(self, window: int = 100) -> dict[str, float]:
        """Get moving averages of all metrics."""
        return {key: self.get_average(key, window) for key in self.metrics}

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step_metrics.clear()


# =============================================================================
# Main Trainer
# =============================================================================


class CrossModalAlignmentTrainer:
    """
    Trainer for cross-modal alignment using contrastive learning.

    This trainer handles:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Logging (local and WandB)
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup loss
        self.loss_fn = InfoNCELoss(
            temperature=config.temperature,
            learnable_temperature=config.temperature_learnable,
        )
        self.loss_fn.to(self.device)

        # Setup optimizer (include temperature if learnable)
        params = list(model.parameters())
        if config.temperature_learnable:
            params.append(self.loss_fn.temperature)

        self.optimizer = AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = None  # Created in train() when we know total steps

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Tracking
        self.metric_tracker = MetricTracker()
        self.checkpoint_manager = CheckpointManager(
            config.output_dir,
            keep_n=config.keep_n_checkpoints,
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode="min" if "loss" in config.early_stopping_metric else "max",
        )

        # WandB
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.to_dict(),
            )

        self.global_step = 0
        self.best_metric = float("inf")

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - self.config.warmup_steps,
            )
        elif self.config.scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=num_training_steps,
                pct_start=self.config.warmup_steps / num_training_steps,
            )

    def _training_step(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute a single training step."""
        # Move batch to device
        batch = batch.to(self.device)

        # Forward pass
        with autocast(enabled=self.config.use_mixed_precision):
            # Get embeddings from model
            # Model should return (embeddings_a, embeddings_b)
            embeddings_a, embeddings_b = self.model(batch)

            # Compute loss
            confidence_weights = (
                batch.confidence_scores if self.config.use_confidence_weighting else None
            )
            loss, metrics = self.loss_fn(
                embeddings_a,
                embeddings_b,
                confidence_weights=confidence_weights,
            )

        return loss, metrics

    def _validation_step(self, batch) -> dict[str, float]:
        """Execute a single validation step."""
        self.model.eval()

        with torch.no_grad():
            batch = batch.to(self.device)
            embeddings_a, embeddings_b = self.model(batch)

            _, metrics = self.loss_fn(
                embeddings_a,
                embeddings_b,
                confidence_weights=None,
            )

        return metrics

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        resume_from: str | None = None,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from
        """
        # Calculate total steps
        steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = min(
            self.config.max_steps,
            steps_per_epoch * 100,  # Max 100 epochs
        )

        # Create scheduler
        self._create_scheduler(num_training_steps)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting training for {num_training_steps} steps")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(
            f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )

        self.model.train()
        accumulated_loss = 0.0

        train_iter = iter(train_dataloader)

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            # Forward pass
            loss, metrics = self._training_step(batch)
            loss = loss / self.config.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # Log accumulated loss
                metrics["loss"] = accumulated_loss
                accumulated_loss = 0.0

            # Update metrics
            self.metric_tracker.update(metrics)
            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics(metrics, prefix="train")

            # Validation
            if val_dataloader and self.global_step % self.config.eval_every_n_steps == 0:
                val_metrics = self._validate(val_dataloader)
                self._log_metrics(val_metrics, prefix="val")

                # Early stopping check
                if self.early_stopping(
                    val_metrics.get(self.config.early_stopping_metric, val_metrics["loss"])
                ):
                    logger.info(f"Early stopping triggered at step {self.global_step}")
                    break

                # Save if best
                if val_metrics["loss"] < self.best_metric:
                    self.best_metric = val_metrics["loss"]
                    self.checkpoint_manager.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.global_step,
                        val_metrics,
                    )

                self.model.train()

            # Periodic checkpoint
            if self.global_step % self.config.save_every_n_steps == 0:
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.global_step,
                    self.metric_tracker.step_metrics,
                )

        # Final checkpoint
        self.checkpoint_manager.save(
            self.model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            self.metric_tracker.step_metrics,
        )

        logger.info(
            f"Training complete. Best model: {self.checkpoint_manager.get_best_checkpoint()}"
        )

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

    def _validate(self, val_dataloader: DataLoader) -> dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        all_metrics = []

        for batch in val_dataloader:
            metrics = self._validation_step(batch)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def _log_metrics(self, metrics: dict[str, float], prefix: str = ""):
        """Log metrics to console and WandB."""
        # Format for console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} [{prefix}] {metrics_str}")

        # WandB
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()},
                step=self.global_step,
            )

    def _load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = self.checkpoint_manager.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["step"]
        logger.info(f"Resumed from checkpoint at step {self.global_step}")


# =============================================================================
# Model Wrapper for Training
# =============================================================================


class CrossModalAlignmentModel(nn.Module):
    """
    Wrapper model for cross-modal alignment training.

    This wraps the modality-specific encoders and cross-modal projector
    into a single module for training.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        spectral_encoder: nn.Module,
        molecular_encoder: nn.Module,
        projector: nn.Module,
    ):
        super().__init__()

        self.encoders = nn.ModuleDict(
            {
                "text_scientific": text_encoder,
                "text_policy": text_encoder,  # Share text encoder
                "text_news": text_encoder,
                "spectrum_xrd": spectral_encoder,
                "spectrum_xrf": spectral_encoder,  # Share spectral encoder
                "spectrum_raman": spectral_encoder,
                "crystal_structure": molecular_encoder,
                "molecular_structure": molecular_encoder,
            }
        )

        self.projector = projector

    def encode_modality(self, content: Any, modality_type: str) -> torch.Tensor:
        """Encode content using the appropriate encoder."""
        encoder = self.encoders.get(modality_type)
        if encoder is None:
            raise ValueError(f"Unknown modality type: {modality_type}")

        embedding = encoder(content)
        return embedding

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a batch.

        Returns projected embeddings for modality A and B.
        """
        # This is a simplified version - actual implementation would need
        # to handle different modality types appropriately

        # Encode modality A
        embeddings_a = []
        for content, mod_type in zip(
            batch.modality_a_contents, batch.modality_a_types, strict=False
        ):
            emb = self.encode_modality(content, mod_type)
            embeddings_a.append(emb)
        embeddings_a = torch.stack(embeddings_a)

        # Encode modality B
        embeddings_b = []
        for content, mod_type in zip(
            batch.modality_b_contents, batch.modality_b_types, strict=False
        ):
            emb = self.encode_modality(content, mod_type)
            embeddings_b.append(emb)
        embeddings_b = torch.stack(embeddings_b)

        # Project to shared space
        proj_a = self.projector(embeddings_a, batch.modality_a_types[0])
        proj_b = self.projector(embeddings_b, batch.modality_b_types[0])

        return proj_a, proj_b


# =============================================================================
# Training Script Entry Point
# =============================================================================


def train_cross_modal_alignment(
    corpus_path: str,
    output_dir: str,
    config: TrainingConfig | None = None,
):
    """
    Main entry point for training cross-modal alignment.

    Args:
        corpus_path: Path to training corpus JSONL
        output_dir: Directory for checkpoints and logs
        config: Optional training configuration
    """
    from .paired_data_loader import CMMPairedDataset, create_contrastive_dataloader

    # Default config
    if config is None:
        config = TrainingConfig(output_dir=output_dir)

    # Create datasets
    train_dataset = CMMPairedDataset(
        corpus_path,
        min_confidence=0.3,
        augment=True,
    )

    # Split for validation (simple approach)
    val_size = int(0.1 * len(train_dataset))
    list(range(len(train_dataset) - val_size))
    list(range(len(train_dataset) - val_size, len(train_dataset)))

    # Create dataloaders
    create_contrastive_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        sampler_type="balanced",
    )

    # For validation, create a separate dataset
    val_dataset = CMMPairedDataset(
        corpus_path,
        min_confidence=0.0,
        augment=False,
    )
    create_contrastive_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Create model (placeholder - use actual encoders)
    logger.info("Creating model...")
    # model = CrossModalAlignmentModel(...)

    # Create trainer
    # trainer = CrossModalAlignmentTrainer(model, config)

    # Train
    # trainer.train(train_dataloader, val_dataloader)

    logger.info("Training script ready - implement model creation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train cross-modal alignment")
    parser.add_argument("corpus_path", help="Path to training corpus")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        use_wandb=args.use_wandb,
    )

    train_cross_modal_alignment(args.corpus_path, args.output_dir, config)
