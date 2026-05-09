"""
Main training loop for all models.

Features:
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping
  - AdamW + cosine annealing with warmup
  - Early stopping on validation AUPRC
  - W&B logging (optional)
  - Multi-seed experiment management
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.training.losses import FocalLoss, MultiTaskLoss
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.config_loader import get

logger = logging.getLogger(__name__)


def _warmup_lr(optimizer, step: int, warmup_steps: int, base_lr: float) -> None:
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def _compute_val_auprc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Quick validation AUPRC for early stopping."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["m"].to(device)
            delta_t = batch["delta_t"].to(device)
            s = batch["s"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            y = batch["y"]

            out = model(x, m, delta_t, s, attn_mask)
            probs = torch.sigmoid(out["logit_sepsis"]).squeeze(-1)  # (B, T)
            mask = batch["attention_mask"]

            # Flatten valid positions
            probs_flat = probs[mask].cpu().numpy()
            labels_flat = y[mask].numpy()
            all_probs.append(probs_flat)
            all_labels.append(labels_flat)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Filter out NaN labels or NaN predictions
    valid = ~np.isnan(all_labels) & ~np.isnan(all_probs)
    all_probs = all_probs[valid]
    all_labels = all_labels[valid]

    if len(all_labels) == 0 or all_labels.sum() == 0:
        return 0.0
    return average_precision_score(all_labels, all_probs)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    model_name: str,
    save_dir: Path,
    device: torch.device,
    use_multi_task: bool = False,
    seed: int = 42,
) -> dict:
    """
    Train a model and return training history.

    Args:
        model:          PyTorch model (must accept x, m, delta_t, s, attn_mask)
        train_loader:   training DataLoader
        val_loader:     validation DataLoader
        cfg:            loaded config dict
        model_name:     name for checkpointing (e.g. "imst_mamba")
        save_dir:       root directory for checkpoints and logs
        device:         torch.device
        use_multi_task: use MultiTaskLoss (for IMST-Mamba)
        seed:           random seed (for naming)

    Returns:
        dict with training history
    """
    t_cfg = cfg["training"]
    max_epochs = t_cfg["max_epochs"]
    grad_clip = t_cfg.get("grad_clip_norm", 1.0)
    warmup_steps = t_cfg.get("warmup_steps", 500)
    base_lr = t_cfg["optimizer"]["lr"]
    weight_decay = t_cfg["optimizer"].get("weight_decay", 1e-4)
    patience = t_cfg.get("early_stopping_patience", 10)

    # Loss
    loss_cfg = t_cfg.get("loss", {})
    if use_multi_task:
        criterion = MultiTaskLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=loss_cfg.get("alpha", 0.75),
            lambda_mortality=loss_cfg.get("lambda_mortality", 0.3),
            lambda_sofa=loss_cfg.get("lambda_sofa", 0.1),
        )
    else:
        criterion = FocalLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=loss_cfg.get("alpha", 0.75),
        )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay,
                      betas=tuple(t_cfg["optimizer"].get("betas", [0.9, 0.999])))

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=t_cfg["scheduler"].get("eta_min", 1e-6),
    )

    # AMP
    use_amp = (device.type == "cuda") and (cfg.get("precision", "float32") == "float16")
    scaler = GradScaler("cuda", enabled=use_amp)

    # Callbacks
    early_stop = EarlyStopping(patience=patience)
    checkpoint = ModelCheckpoint(
        save_dir / f"seed_{seed}",
        model_name=f"{model_name}_s{seed}",
    )

    # W&B (optional)
    use_wandb = get(cfg, "logging.use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=get(cfg, "logging.project_name", "sepsis-imst-mamba"),
            name=f"{model_name}_seed{seed}",
            config=cfg,
        )

    history = {"train_loss": [], "val_auprc": [], "lr": []}
    global_step = 0

    model = model.to(device)

    logger.info(f"Training {model_name} | seed={seed} | device={device} | amp={use_amp}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch in train_loader:
            # Warmup LR
            _warmup_lr(optimizer, global_step, warmup_steps, base_lr)
            global_step += 1

            # Move to device (single copy)
            batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
            x        = batch_gpu["x"]
            m        = batch_gpu["m"]
            delta_t  = batch_gpu["delta_t"]
            s        = batch_gpu["s"]
            attn_mask = batch_gpu["attention_mask"]

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                outputs = model(x, m, delta_t, s, attn_mask)

                if use_multi_task:
                    losses = criterion(outputs, batch_gpu)
                    loss = losses["total"]
                else:
                    logit = outputs["logit_sepsis"].squeeze(-1)
                    loss = criterion(
                        logit,
                        batch_gpu["y"],
                        mask=batch_gpu["attention_mask"],
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            del outputs, loss, batch_gpu, x, m, delta_t, s, attn_mask
            if global_step % 50 == 0:
                import gc; gc.collect()

        # Scheduler step (after warmup)
        if global_step >= warmup_steps:
            scheduler.step()

        avg_loss = np.mean(epoch_losses)
        val_auprc = _compute_val_auprc(model, val_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["val_auprc"].append(val_auprc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_auprc={val_auprc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": avg_loss,
                       "val_auprc": val_auprc, "lr": optimizer.param_groups[0]["lr"]})

        checkpoint.step(val_auprc, model, epoch)
        checkpoint.save_last(model, epoch)

        if early_stop.step(val_auprc):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if checkpoint.best_path is not None:
        checkpoint.load_best(model)

    if use_wandb:
        import wandb
        wandb.finish()

    return history
