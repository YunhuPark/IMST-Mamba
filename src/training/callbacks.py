"""
Training callbacks: early stopping and model checkpointing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when a metric hasn't improved for `patience` epochs.
    Higher is better (e.g., AUPRC).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.counter = 0
        self.stopped = False

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


class ModelCheckpoint:
    """Save the best model checkpoint by validation metric."""

    def __init__(self, save_dir: Path, model_name: str, metric: str = "auprc"):
        self.save_dir = Path(save_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.metric = metric
        self.best_score = float("-inf")
        self.best_path: Path | None = None

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Save checkpoint if score improved.
        Returns True if a new best was saved.
        """
        if score > self.best_score:
            self.best_score = score
            path = self.save_dir / f"{self.model_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                f"val_{self.metric}": score,
            }, path)
            self.best_path = path
            logger.info(f"Checkpoint saved → {path}  ({self.metric}={score:.4f})")
            return True
        return False

    def save_last(self, model: nn.Module, epoch: int) -> None:
        path = self.save_dir / f"{self.model_name}_last.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }, path)

    def load_best(self, model: nn.Module) -> nn.Module:
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError("No checkpoint saved yet.")
        ckpt = torch.load(self.best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
        return model
