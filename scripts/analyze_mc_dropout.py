"""
B2: MC Dropout uncertainty quantification for IMST-Mamba.

Runs N stochastic forward passes with dropout active at inference time.
Computes:
  1. Mean prediction and epistemic uncertainty (std dev across passes)
  2. Uncertainty vs prediction accuracy (calibration)
  3. Uncertainty vs missingness rate (model knows when to be uncertain)
  4. Expected Calibration Error (ECE)
  5. Reliability diagram data

Usage:
    python scripts/analyze_mc_dropout.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import full_metrics, print_metrics
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MC_PASSES = 10   # number of stochastic forward passes (10 sufficient for uncertainty est.)
N_ECE_BINS = 10


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep model in eval mode but re-enable Dropout layers for MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def reliability_diagram_data(y_true: np.ndarray, probs: np.ndarray,
                              n_bins: int = 10) -> list[dict]:
    """Bin data for reliability diagram."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bins.append({"conf": float((lo + hi) / 2), "acc": None, "n": 0})
        else:
            bins.append({
                "conf": float(probs[mask].mean()),
                "acc":  float(y_true[mask].mean()),
                "n":    int(mask.sum()),
            })
    return bins


def main():
    cfg       = load_config("configs/base.yaml")
    model_cfg = load_config("configs/base.yaml", "configs/model_imst_mamba.yaml")
    save_dir  = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info(f"Device: {device}  MC passes: {MC_PASSES}")

    ckpt_paths = sorted(save_dir.glob("*/checkpoints/imst_mamba*_best.pt"))
    if not ckpt_paths:
        logger.error("No IMST-Mamba checkpoint found.")
        return

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    # ── Run MC Dropout inference ──────────────────────────────────────────
    from src.models.imst_mamba import build_model

    # We use the first checkpoint (or average over seeds below)
    all_pass_probs = []   # list of length MC_PASSES × n_seeds; each entry is flat array
    y_true_ref     = None
    miss_rate_ref  = None

    for ckpt_path in ckpt_paths:
        model = build_model(model_cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        enable_mc_dropout(model)
        logger.info(f"MC Dropout on {ckpt_path.name}")

        for pass_i in range(MC_PASSES):
            probs_list, labels_list, miss_list = [], [], []

            with torch.no_grad():
                for batch in test_loader:
                    xb   = batch["x"].to(device)
                    mb   = batch["m"].to(device)
                    dtb  = batch["delta_t"].to(device)
                    sb   = batch["s"].to(device)
                    attn = batch["attention_mask"].to(device)
                    y    = batch["y"]
                    attn_cpu = batch["attention_mask"]
                    m_cpu    = batch["m"]

                    out = model(xb, mb, dtb, sb, attn)
                    p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()

                    for i in range(xb.shape[0]):
                        T = int(attn_cpu[i].sum())
                        y_i = y[i, :T].numpy()
                        valid = ~np.isnan(y_i)
                        if valid.sum() == 0:
                            continue
                        probs_list.append(p[i, :T].numpy()[valid])
                        labels_list.append(y_i[valid])
                        mr = float(1 - m_cpu[i, :T].mean())
                        miss_list.append(np.full(valid.sum(), mr))

            pass_probs = np.concatenate(probs_list)
            all_pass_probs.append(pass_probs)

            if y_true_ref is None:
                y_true_ref   = np.concatenate(labels_list)
                miss_rate_ref = np.concatenate(miss_list)

            if pass_i % 10 == 0:
                logger.info(f"  Pass {pass_i+1}/{MC_PASSES}")

        del model

    if y_true_ref is None:
        logger.error("No data collected.")
        return

    # Stack: (n_seeds × MC_PASSES, N_timesteps)
    all_pass_probs = np.stack(all_pass_probs, axis=0)   # (K, N)
    mean_probs  = all_pass_probs.mean(axis=0)            # (N,)
    std_probs   = all_pass_probs.std(axis=0)             # (N,) = epistemic uncertainty
    y_true = y_true_ref

    logger.info(f"Timesteps: {len(y_true):,}  "
                f"Uncertainty mean={std_probs.mean():.4f}  max={std_probs.max():.4f}")

    # ── Deterministic metrics (mean pred) ────────────────────────────────
    logger.info("Computing metrics on mean MC predictions...")
    r_mc = full_metrics(y_true, mean_probs, n_bootstrap=200)
    print_metrics(r_mc, "IMST-Mamba (MC mean)")

    # ── ECE and calibration ───────────────────────────────────────────────
    ece = compute_ece(y_true, mean_probs, N_ECE_BINS)
    logger.info(f"ECE: {ece:.4f}")
    rel_diag = reliability_diagram_data(y_true, mean_probs, N_ECE_BINS)

    # ── Uncertainty vs accuracy ───────────────────────────────────────────
    # Sort timesteps into uncertainty deciles and compute accuracy per decile
    unc_deciles = np.percentile(std_probs, np.linspace(0, 100, 11))
    uncertainty_vs_acc = []
    for lo, hi in zip(unc_deciles[:-1], unc_deciles[1:]):
        mask = (std_probs >= lo) & (std_probs <= hi)
        if mask.sum() == 0:
            continue
        acc  = float(((mean_probs[mask] >= 0.5) == y_true[mask]).mean())
        auroc_bin = None
        if len(np.unique(y_true[mask])) == 2:
            from sklearn.metrics import roc_auc_score
            try:
                auroc_bin = float(roc_auc_score(y_true[mask], mean_probs[mask]))
            except Exception:
                pass
        uncertainty_vs_acc.append({
            "unc_lo":  float(lo),
            "unc_hi":  float(hi),
            "mean_unc": float(std_probs[mask].mean()),
            "accuracy": acc,
            "auroc":    auroc_bin,
            "n":        int(mask.sum()),
        })

    # ── Uncertainty vs missingness rate ──────────────────────────────────
    # Bin by missingness rate and show mean uncertainty
    miss_deciles = np.percentile(miss_rate_ref, np.linspace(0, 100, 11))
    uncertainty_vs_miss = []
    for lo, hi in zip(miss_deciles[:-1], miss_deciles[1:]):
        mask = (miss_rate_ref >= lo) & (miss_rate_ref <= hi)
        if mask.sum() == 0:
            continue
        uncertainty_vs_miss.append({
            "miss_lo":    float(lo),
            "miss_hi":    float(hi),
            "mean_miss":  float(miss_rate_ref[mask].mean()),
            "mean_unc":   float(std_probs[mask].mean()),
            "n":          int(mask.sum()),
        })

    logger.info("Uncertainty vs missingness (first 5 deciles):")
    for d in uncertainty_vs_miss[:5]:
        logger.info(f"  miss={d['mean_miss']:.3f}  unc={d['mean_unc']:.4f}")

    # ── High-uncertainty analysis ──────────────────────────────────────────
    high_unc_thresh = np.percentile(std_probs, 90)
    high_unc_mask   = std_probs >= high_unc_thresh

    logger.info(f"High-uncertainty (top 10%) timesteps: {high_unc_mask.sum():,}")
    logger.info(f"  Sepsis prevalence in high-unc: {y_true[high_unc_mask].mean():.4f}")
    logger.info(f"  Sepsis prevalence overall:     {y_true.mean():.4f}")

    high_unc_stats = {
        "threshold_90pct":  float(high_unc_thresh),
        "sepsis_prev_high_unc": float(y_true[high_unc_mask].mean()),
        "sepsis_prev_overall":  float(y_true.mean()),
        "miss_rate_high_unc":   float(miss_rate_ref[high_unc_mask].mean()),
        "miss_rate_overall":    float(miss_rate_ref.mean()),
    }

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "mc_passes":         MC_PASSES,
        "mc_metrics":        r_mc,
        "ece":               ece,
        "reliability_diagram": rel_diag,
        "uncertainty_vs_accuracy":   uncertainty_vs_acc,
        "uncertainty_vs_missingness": uncertainty_vs_miss,
        "high_uncertainty_analysis": high_unc_stats,
    }

    out_path = save_dir / "mc_dropout_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    logger.info(f"Saved → {out_path}")

    print(f"\nMC Dropout Summary (K={len(ckpt_paths)*MC_PASSES} passes):")
    print(f"  AUROC = {r_mc['auroc']:.4f}  AUPRC = {r_mc['auprc']:.4f}")
    print(f"  ECE   = {ece:.4f}")
    print(f"  Mean uncertainty (std) = {std_probs.mean():.4f}")
    print(f"  Uncertainty corr w/ miss rate: "
          f"{float(np.corrcoef(miss_rate_ref, std_probs)[0,1]):.4f}")


if __name__ == "__main__":
    main()
