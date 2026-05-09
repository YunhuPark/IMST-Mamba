"""
B4: Imputation strategy comparison — Transformer vs IMST-Mamba.

Tests the Transformer with three inference-time imputation variants:
  1. zero_imp:   unobserved → 0 (≈ mean in normalized space; current training setup)
  2. ffill_imp:  unobserved → carry-forward last observed value
  3. no_mask:    same as zero_imp but mask channel set to 0 (model "blind" to missingness)

Compares all variants against IMST-Mamba (no imputation needed) across:
  - Overall AUROC / AUPRC
  - High-missingness subgroup (top-33%)
  - Low-missingness subgroup (bottom-33%)

Motivation: shows IMST-Mamba is robust without a separate imputation step.

Usage:
    python scripts/analyze_imputation_comparison.py
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


def ffill_batch(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Forward-fill missing values along the time axis.
    x: (B, T, F), m: (B, T, F) — returns x_filled (B, T, F).
    Positions with m=0 and no prior observation remain 0.
    """
    B, T, F = x.shape
    x_out = x.clone()
    for b in range(B):
        last_val = torch.zeros(F, device=x.device)
        for t in range(T):
            # Update last_val where observed
            obs = m[b, t] > 0.5           # (F,)
            last_val = torch.where(obs, x[b, t], last_val)
            # Fill missing positions
            x_out[b, t] = torch.where(obs, x[b, t], last_val)
    return x_out


def run_transformer_inference(model, test_loader, device,
                               imputation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Transformer inference with a given imputation strategy.

    Returns: (y_true, probs, miss_rates) — all per valid timestep.
    """
    model.eval()
    probs_list, labels_list, miss_list = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x    = batch["x"].to(device)        # (B, T, F)
            m    = batch["m"].to(device)
            dt   = batch["delta_t"].to(device)
            s    = batch["s"].to(device)
            attn = batch["attention_mask"].to(device)
            y    = batch["y"]
            attn_cpu = batch["attention_mask"]
            m_cpu    = batch["m"]

            # Apply imputation strategy
            if imputation == "ffill_imp":
                x_inp = ffill_batch(x, m)
                m_inp = m
            elif imputation == "no_mask":
                x_inp = x                         # zero where unobserved
                m_inp = torch.zeros_like(m)       # hide mask from model
            else:                                  # zero_imp (default)
                x_inp = x
                m_inp = m

            out = model(x_inp, m_inp, dt, s, attn)
            p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()

            for i in range(x.shape[0]):
                T = int(attn_cpu[i].sum())
                y_i = y[i, :T].numpy()
                valid = ~np.isnan(y_i)
                if valid.sum() == 0:
                    continue
                probs_list.append(p[i, :T].numpy()[valid])
                labels_list.append(y_i[valid])
                mr = float(1 - m_cpu[i, :T].mean())
                miss_list.append(np.full(valid.sum(), mr))

    return (np.concatenate(labels_list),
            np.concatenate(probs_list),
            np.concatenate(miss_list))


def run_imst_inference(model, test_loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_list, labels_list, miss_list = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x    = batch["x"].to(device)
            m    = batch["m"].to(device)
            dt   = batch["delta_t"].to(device)
            s    = batch["s"].to(device)
            attn = batch["attention_mask"].to(device)
            y    = batch["y"]
            attn_cpu = batch["attention_mask"]
            m_cpu    = batch["m"]

            out = model(x, m, dt, s, attn)
            p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()

            for i in range(x.shape[0]):
                T = int(attn_cpu[i].sum())
                y_i = y[i, :T].numpy()
                valid = ~np.isnan(y_i)
                if valid.sum() == 0:
                    continue
                probs_list.append(p[i, :T].numpy()[valid])
                labels_list.append(y_i[valid])
                mr = float(1 - m_cpu[i, :T].mean())
                miss_list.append(np.full(valid.sum(), mr))

    return (np.concatenate(labels_list),
            np.concatenate(probs_list),
            np.concatenate(miss_list))


def metrics_by_missingness(y: np.ndarray, probs: np.ndarray,
                            miss: np.ndarray) -> dict:
    """Overall + low/high missingness subgroup metrics."""
    p33 = np.percentile(miss, 33)
    p66 = np.percentile(miss, 66)
    groups = {
        "overall":   np.ones(len(y), dtype=bool),
        "miss_low":  miss <= p33,
        "miss_high": miss > p66,
    }
    out = {}
    for gname, mask in groups.items():
        y_g, p_g = y[mask], probs[mask]
        if len(np.unique(y_g)) < 2:
            continue
        r = full_metrics(y_g, p_g, n_bootstrap=100)
        out[gname] = r
    return out


def main():
    cfg       = load_config("configs/base.yaml")
    model_cfg_imst = load_config("configs/base.yaml", "configs/model_imst_mamba.yaml")
    model_cfg_tr   = load_config("configs/base.yaml", "configs/model_transformer.yaml")
    save_dir  = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info(f"Device: {device}")

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    results = {}

    # ── IMST-Mamba (reference) ────────────────────────────────────────────
    from src.models.imst_mamba import build_model as build_imst
    imst_ckpts = sorted(save_dir.glob("*/checkpoints/imst_mamba*_best.pt"))
    if imst_ckpts:
        logger.info("\n=== IMST-Mamba ===")
        seed_preds = []
        y_ref = miss_ref = None

        for ckpt_path in imst_ckpts:
            model = build_imst(model_cfg_imst).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            y_t, p_t, mr_t = run_imst_inference(model, test_loader, device)
            seed_preds.append(p_t)
            if y_ref is None:
                y_ref = y_t; miss_ref = mr_t
            del model

        avg_p = np.mean(seed_preds, axis=0)
        results["IMST-Mamba"] = metrics_by_missingness(y_ref, avg_p, miss_ref)
        print_metrics(results["IMST-Mamba"]["overall"], "IMST-Mamba (overall)")
        logger.info(f"  miss_low  AUROC={results['IMST-Mamba']['miss_low']['auroc']:.4f}")
        logger.info(f"  miss_high AUROC={results['IMST-Mamba']['miss_high']['auroc']:.4f}")

    # ── Transformer variants ──────────────────────────────────────────────
    from src.models.transformer_baseline import build_model as build_tr
    tr_ckpts = sorted(save_dir.glob("*/checkpoints/transformer*_best.pt"))

    if tr_ckpts:
        for imputation in ["zero_imp", "ffill_imp", "no_mask"]:
            label = f"Transformer ({imputation})"
            logger.info(f"\n=== {label} ===")

            seed_preds = []
            y_ref_tr = miss_ref_tr = None

            for ckpt_path in tr_ckpts:
                model = build_tr(model_cfg_tr).to(device)
                ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                y_t, p_t, mr_t = run_transformer_inference(
                    model, test_loader, device, imputation)
                seed_preds.append(p_t)
                if y_ref_tr is None:
                    y_ref_tr = y_t; miss_ref_tr = mr_t
                del model

            avg_p = np.mean(seed_preds, axis=0)
            results[label] = metrics_by_missingness(y_ref_tr, avg_p, miss_ref_tr)
            print_metrics(results[label]["overall"], f"{label} (overall)")
            if "miss_low" in results[label]:
                logger.info(f"  miss_low  AUROC={results[label]['miss_low']['auroc']:.4f}")
            if "miss_high" in results[label]:
                logger.info(f"  miss_high AUROC={results[label]['miss_high']['auroc']:.4f}")
    else:
        logger.warning("No Transformer checkpoints found.")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = save_dir / "imputation_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    logger.info(f"\nSaved → {out_path}")

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "="*85)
    print(f"{'Method':<35} {'Overall AUROC':>13} {'Low Miss AUROC':>14} {'High Miss AUROC':>15}")
    print("="*85)
    for name, group_res in results.items():
        ov  = group_res.get("overall",   {}).get("auroc", float("nan"))
        lo  = group_res.get("miss_low",  {}).get("auroc", float("nan"))
        hi  = group_res.get("miss_high", {}).get("auroc", float("nan"))
        print(f"{name:<35} {ov:>13.4f} {lo:>14.4f} {hi:>15.4f}")
    print("="*85)
    print("Higher AUROC in 'High Miss' = better missingness handling")


if __name__ == "__main__":
    main()
