"""
Missingness visualization analysis.

Extracts:
  1. Feature-level missingness rates in test set
  2. Per-patient missingness state distributions
  3. Attention weights / feature importance (if model supports it)
  4. Correlation between missingness rate and prediction confidence

Usage:
    python scripts/analyze_missingness_viz.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]


def main():
    cfg = load_config("configs/base.yaml")
    model_cfg = load_config("configs/base.yaml", "configs/model_imst_mamba.yaml")
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    # ── 1. Feature-level missingness rates ───────────────────────────────
    logger.info("Computing feature missingness rates...")
    feat_obs   = np.zeros(34)
    feat_total = np.zeros(34)
    miss_state_counts = np.zeros((34, 3))  # (F, state)

    for batch in test_loader:
        m    = batch["m"]          # (B, T, F)
        ms   = batch["miss_state"] # (B, T, F)
        attn = batch["attention_mask"]  # (B, T)

        for i in range(m.shape[0]):
            T = int(attn[i].sum())
            feat_obs   += m[i, :T].sum(0).numpy()
            feat_total += T
            for state in range(3):
                miss_state_counts[:, state] += (ms[i, :T] == state).float().sum(0).numpy()

    obs_rate = feat_obs / (feat_total + 1e-10)
    miss_state_frac = miss_state_counts / (miss_state_counts.sum(1, keepdims=True) + 1e-10)

    feat_stats = []
    for i, name in enumerate(FEATURE_NAMES):
        feat_stats.append({
            "feature":        name,
            "obs_rate":       float(obs_rate[i]),
            "miss_rate":      float(1 - obs_rate[i]),
            "state_never":    float(miss_state_frac[i, 0]),
            "state_recent":   float(miss_state_frac[i, 1]),
            "state_stale":    float(miss_state_frac[i, 2]),
        })
    feat_stats.sort(key=lambda x: x["miss_rate"], reverse=True)

    logger.info("Top-10 most-missing features:")
    for fs in feat_stats[:10]:
        logger.info(f"  {fs['feature']:<20} miss={fs['miss_rate']:.3f}  "
                    f"never={fs['state_never']:.3f}  "
                    f"stale={fs['state_stale']:.3f}")

    # ── 2. Missingness state dist per sepsis vs non-sepsis ──────────────
    logger.info("Computing missingness state by sepsis label...")
    state_by_label = {0: np.zeros((34, 3)), 1: np.zeros((34, 3))}
    count_by_label = {0: 0, 1: 0}

    for batch in test_loader:
        ms   = batch["miss_state"]  # (B, T, F)
        y    = batch["y"]           # (B, T)
        attn = batch["attention_mask"]

        for i in range(ms.shape[0]):
            T = int(attn[i].sum())
            y_valid = y[i, :T]
            y_valid = y_valid[~torch.isnan(y_valid)]
            label = int(y_valid.max().item()) if len(y_valid) > 0 else 0
            for state in range(3):
                state_by_label[label][:, state] += (ms[i, :T] == state).float().sum(0).numpy()
            count_by_label[label] += T

    state_by_label_norm = {}
    for lbl in [0, 1]:
        total = state_by_label[lbl].sum(1, keepdims=True) + 1e-10
        state_by_label_norm[lbl] = (state_by_label[lbl] / total).tolist()

    # ── 3. IMST-Mamba inference: prediction confidence vs missingness ────
    logger.info("Running IMST-Mamba inference for confidence analysis...")
    ckpt_paths = sorted(save_dir.glob("*/checkpoints/imst_mamba*_best.pt"))

    conf_analysis = []
    if ckpt_paths:
        from src.models.imst_mamba import build_model
        model = build_model(model_cfg).to(device)
        ckpt  = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                x       = batch["x"].to(device)
                m_gpu   = batch["m"].to(device)
                delta_t = batch["delta_t"].to(device)
                s       = batch["s"].to(device)
                attn    = batch["attention_mask"].to(device)
                y       = batch["y"]
                attn_cpu= batch["attention_mask"]
                m_cpu   = batch["m"]

                out = model(x, m_gpu, delta_t, s, attn)
                p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()

                for i in range(x.shape[0]):
                    T = int(attn_cpu[i].sum())
                    miss_r  = float(1 - m_cpu[i, :T].mean())
                    conf_r  = float(p[i, :T].mean())
                    y_valid = y[i, :T]
                    y_valid = y_valid[~torch.isnan(y_valid)]
                    sepsis  = int(y_valid.max().item()) if len(y_valid) > 0 else 0
                    conf_analysis.append({
                        "miss_rate": miss_r,
                        "mean_prob": conf_r,
                        "has_sepsis": sepsis,
                    })
        del model

    # ── Save all results ─────────────────────────────────────────────────
    viz_results = {
        "feature_stats":        feat_stats,
        "miss_state_by_label":  state_by_label_norm,
        "confidence_analysis":  conf_analysis,
        "feature_names":        FEATURE_NAMES,
    }

    out_path = save_dir / "missingness_viz.json"
    with open(out_path, "w") as f:
        json.dump(viz_results, f, indent=2)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
