"""
Ablation study — inference-time component disabling.

Tests IMST-Mamba with each key component removed:
  - full:          Full IMST-Mamba (baseline)
  - no_miss_state: missingness state zeroed out (→ 0=never everywhere)
  - no_s:          time-since-last-obs zeroed out
  - no_mask:       observation mask zeroed out (treat all as observed)
  - no_delta_t:    inter-event gaps set to constant 1h

Usage:
    python scripts/analyze_ablation.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import full_metrics, print_metrics
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ablation_inference(model, test_loader, device, ablation: str):
    """
    Run inference with one component disabled.

    ablation options:
        "full"          — normal forward pass
        "no_miss_state" — zero out miss_state in model (if model uses it)
        "no_s"          — zero out s (time-since-last-obs)
        "no_mask"       — set m to all-ones (pretend all observed)
        "no_delta_t"    — set delta_t to constant 1h
    """
    model.eval()
    probs_list, labels_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            x       = batch["x"].to(device)
            m       = batch["m"].to(device)
            delta_t = batch["delta_t"].to(device)
            s       = batch["s"].to(device)
            attn    = batch["attention_mask"].to(device)
            y       = batch["y"]
            mask    = batch["attention_mask"]

            # Apply ablation
            if ablation == "no_mask":
                m = torch.ones_like(m)
            elif ablation == "no_s":
                s = torch.zeros_like(s)
            elif ablation == "no_delta_t":
                delta_t = torch.ones_like(delta_t)  # constant 1h

            out = model(x, m, delta_t, s, attn)
            p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()

            probs_list.append(p[mask].numpy())
            labels_list.append(y[mask].numpy())

    return np.concatenate(probs_list), np.concatenate(labels_list)


def main():
    cfg      = load_config("configs/base.yaml")
    model_cfg = load_config("configs/base.yaml", "configs/model_imst_mamba.yaml")
    save_dir  = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info(f"Device: {device}")

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    # Load IMST-Mamba checkpoints
    from src.models.imst_mamba import build_model
    ckpt_paths = sorted(save_dir.glob("*/checkpoints/imst_mamba*_best.pt"))
    if not ckpt_paths:
        logger.error("No IMST-Mamba checkpoints found")
        return

    ablations = ["full", "no_mask", "no_s", "no_delta_t"]
    results = {}

    for ablation in ablations:
        logger.info(f"\n--- Ablation: {ablation} ---")
        seed_probs = []
        y_true_ref = None

        for ckpt_path in ckpt_paths:
            model = build_model(model_cfg).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            probs, labels = run_ablation_inference(model, test_loader, device, ablation)
            seed_probs.append(probs)
            if y_true_ref is None:
                y_true_ref = labels
            del model

        avg_probs = np.mean(seed_probs, axis=0)
        r = full_metrics(y_true_ref, avg_probs, n_bootstrap=500)
        results[ablation] = r
        print_metrics(r, f"IMST-Mamba [{ablation}]")

    # Save
    out_path = save_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    logger.info(f"Saved → {out_path}")

    # Print delta table
    print("\n" + "="*60)
    print(f"{'Ablation':<20} {'AUROC':>8} {'ΔAUROC':>8} {'AUPRC':>8} {'ΔAUPRC':>8}")
    print("="*60)
    base_auroc = results["full"]["auroc"]
    base_auprc = results["full"]["auprc"]
    for abl, r in results.items():
        delta_auroc = r["auroc"] - base_auroc
        delta_auprc = r["auprc"] - base_auprc
        print(f"{abl:<20} {r['auroc']:>8.4f} {delta_auroc:>+8.4f} "
              f"{r['auprc']:>8.4f} {delta_auprc:>+8.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
