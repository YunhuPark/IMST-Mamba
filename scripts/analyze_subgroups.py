"""
Subgroup analysis for top-tier paper.

Analyses:
  1. Missingness tier (low / mid / high missing rate)
  2. Early-prediction windows (first 6h / 12h / 24h of stay)
  3. Lab-only missingness (high lab-missing patients)
  4. Sequence length (short / medium / long stays)

Usage:
    python scripts/analyze_subgroups.py
"""
from __future__ import annotations

import importlib
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

MODEL_REGISTRY = {
    "imst_mamba": ("src.models.imst_mamba", "build_model"),
    "grud":       ("src.models.grud", "build_model"),
    "lstm":       ("src.models.lstm_baseline", "build_model"),
    "transformer":("src.models.transformer_baseline", "build_model"),
    "retain":     ("src.models.retain", "build_model"),
}
CONFIG_REGISTRY = {
    "imst_mamba": "configs/model_imst_mamba.yaml",
    "grud":       "configs/model_grud.yaml",
    "lstm":       "configs/model_lstm.yaml",
    "transformer":"configs/model_transformer.yaml",
    "retain":     None,
}
LAB_FEATURE_INDICES = list(range(8, 34))   # indices 8-33 are lab features


# ---------------------------------------------------------------------------
# Inference with per-patient tracking
# ---------------------------------------------------------------------------

def run_inference_per_patient(model, test_loader, device):
    """
    Returns dict: stay_id -> {
        probs:    np.ndarray (T,)   -- sigmoid probabilities
        labels:   np.ndarray (T,)
        m:        np.ndarray (T, F) -- observation mask
        delta_t:  np.ndarray (T,)   -- time gap in hours (already normalized)
        seq_len:  int
    }
    """
    model.eval()
    patient_data = {}

    with torch.no_grad():
        for batch in test_loader:
            x       = batch["x"].to(device)
            m_cpu   = batch["m"]                 # keep on CPU for analysis
            m       = m_cpu.to(device)
            delta_t_cpu = batch["delta_t"]
            delta_t = delta_t_cpu.to(device)
            s       = batch["s"].to(device)
            attn    = batch["attention_mask"].to(device)
            y       = batch["y"]
            stay_ids= batch["stay_ids"]

            out = model(x, m, delta_t, s, attn)
            p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()  # (B, T)
            attn_cpu = batch["attention_mask"]

            for i, sid in enumerate(stay_ids):
                T = int(attn_cpu[i].sum().item())
                patient_data[sid] = {
                    "probs":   p[i, :T].numpy(),
                    "labels":  y[i, :T].numpy(),
                    "m":       m_cpu[i, :T].numpy(),       # (T, F)
                    "delta_t": delta_t_cpu[i, :T].numpy(), # (T,) in hours
                    "seq_len": T,
                }

    return patient_data


def collect_model_preds(model_name, cfg, save_dir, test_loader, device):
    """Load all checkpoints for a model and return averaged per-patient data."""
    ckpt_paths = sorted(save_dir.glob(f"*/checkpoints/{model_name}*_best.pt"))
    if not ckpt_paths:
        logger.warning(f"No checkpoint for {model_name}")
        return None

    mod_path, fn_name = MODEL_REGISTRY[model_name]
    mod = importlib.import_module(mod_path)
    build_fn = getattr(mod, fn_name)

    model_cfg_path = CONFIG_REGISTRY.get(model_name)
    if model_cfg_path and Path(model_cfg_path).exists():
        model_cfg = load_config("configs/base.yaml", model_cfg_path)
    else:
        model_cfg = cfg

    seed_results = []
    for ckpt_path in ckpt_paths:
        model = build_fn(model_cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded {ckpt_path.name}")
        pd_map = run_inference_per_patient(model, test_loader, device)
        seed_results.append(pd_map)
        del model

    # Average probs across seeds (keep other fields from first seed)
    merged = {}
    for sid in seed_results[0]:
        merged[sid] = dict(seed_results[0][sid])
        if len(seed_results) > 1:
            merged[sid]["probs"] = np.mean(
                [sr[sid]["probs"] for sr in seed_results if sid in sr], axis=0
            )
    return merged


# ---------------------------------------------------------------------------
# Subgroup helpers
# ---------------------------------------------------------------------------

def compute_patient_stats(pd_map):
    """Per-patient summary statistics."""
    stats = {}
    for sid, d in pd_map.items():
        miss_rate     = 1.0 - float(d["m"].mean())
        lab_miss_rate = 1.0 - float(d["m"][:, LAB_FEATURE_INDICES].mean())
        labels_valid  = d["labels"][~np.isnan(d["labels"])]
        has_sepsis    = int(labels_valid.max()) if len(labels_valid) > 0 else 0
        cumtime       = float(np.cumsum(d["delta_t"])[-1])  # total stay hours
        stats[sid] = {
            "miss_rate":     miss_rate,
            "lab_miss_rate": lab_miss_rate,
            "has_sepsis":    has_sepsis,
            "stay_hours":    cumtime,
            "seq_len":       d["seq_len"],
        }
    return stats


def metrics_for_subgroup(pd_map, stay_ids_subset, n_bootstrap=500):
    """Compute metrics on a subset of patients."""
    probs_list, labels_list = [], []
    for sid in stay_ids_subset:
        if sid not in pd_map:
            continue
        probs_list.append(pd_map[sid]["probs"])
        labels_list.append(pd_map[sid]["labels"])

    if not probs_list:
        return None
    y_score = np.concatenate(probs_list)
    y_true  = np.concatenate(labels_list)

    if len(np.unique(y_true)) < 2:
        return None
    return full_metrics(y_true, y_score, n_bootstrap=n_bootstrap)


def metrics_first_K_hours(pd_map, stay_ids_subset, K_hours, n_bootstrap=500):
    """
    Evaluate using only the first K hours of each stay.
    delta_t is already in hours (normalized by dataset.py).
    """
    probs_list, labels_list = [], []
    for sid in stay_ids_subset:
        if sid not in pd_map:
            continue
        d = pd_map[sid]
        cum = np.cumsum(d["delta_t"])
        mask = cum <= K_hours
        if mask.sum() == 0:
            continue
        probs_list.append(d["probs"][mask])
        labels_list.append(d["labels"][mask])

    if not probs_list:
        return None
    y_score = np.concatenate(probs_list)
    y_true  = np.concatenate(labels_list)

    if len(np.unique(y_true)) < 2:
        return None
    return full_metrics(y_true, y_score, n_bootstrap=n_bootstrap)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg      = load_config("configs/base.yaml")
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info(f"Device: {device}")

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    results = {}

    for model_name in MODEL_REGISTRY:
        logger.info(f"\n{'='*50}\nAnalysing {model_name}\n{'='*50}")
        pd_map = collect_model_preds(model_name, cfg, save_dir, test_loader, device)
        if pd_map is None:
            continue

        pt_stats = compute_patient_stats(pd_map)
        all_sids = list(pd_map.keys())

        # ── 1. Missingness tiers ──────────────────────────────────────────
        miss_vals = [pt_stats[s]["miss_rate"] for s in all_sids]
        p33 = np.percentile(miss_vals, 33)
        p66 = np.percentile(miss_vals, 66)
        low_m  = [s for s in all_sids if pt_stats[s]["miss_rate"] <= p33]
        mid_m  = [s for s in all_sids if p33 < pt_stats[s]["miss_rate"] <= p66]
        high_m = [s for s in all_sids if pt_stats[s]["miss_rate"] > p66]

        # ── 2. Lab missingness tiers ──────────────────────────────────────
        lab_vals = [pt_stats[s]["lab_miss_rate"] for s in all_sids]
        lab_p66  = np.percentile(lab_vals, 66)
        high_lab_m = [s for s in all_sids if pt_stats[s]["lab_miss_rate"] > lab_p66]
        low_lab_m  = [s for s in all_sids if pt_stats[s]["lab_miss_rate"] <= lab_p66]

        # ── 3. Sequence length ────────────────────────────────────────────
        seqlen_vals = [pt_stats[s]["seq_len"] for s in all_sids]
        sq_p33 = np.percentile(seqlen_vals, 33)
        sq_p66 = np.percentile(seqlen_vals, 66)
        short_stays = [s for s in all_sids if pt_stats[s]["seq_len"] <= sq_p33]
        long_stays  = [s for s in all_sids if pt_stats[s]["seq_len"] > sq_p66]

        subgroup_results = {}

        # Overall
        subgroup_results["overall"] = metrics_for_subgroup(pd_map, all_sids)

        # Missingness tiers
        for label, ids in [("miss_low", low_m), ("miss_mid", mid_m), ("miss_high", high_m)]:
            r = metrics_for_subgroup(pd_map, ids)
            subgroup_results[label] = r
            if r:
                logger.info(f"  {label} (n={len(ids)}): "
                            f"AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

        # Lab missingness
        for label, ids in [("lab_miss_low", low_lab_m), ("lab_miss_high", high_lab_m)]:
            r = metrics_for_subgroup(pd_map, ids)
            subgroup_results[label] = r
            if r:
                logger.info(f"  {label} (n={len(ids)}): "
                            f"AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

        # Sequence length
        for label, ids in [("short_stay", short_stays), ("long_stay", long_stays)]:
            r = metrics_for_subgroup(pd_map, ids)
            subgroup_results[label] = r
            if r:
                logger.info(f"  {label} (n={len(ids)}): "
                            f"AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

        # Early prediction windows
        for K in [6, 12, 24, 36]:
            r = metrics_first_K_hours(pd_map, all_sids, K_hours=K)
            subgroup_results[f"first_{K}h"] = r
            if r:
                logger.info(f"  first_{K}h: "
                            f"AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

        results[model_name] = subgroup_results

    # Save
    out_path = save_dir / "subgroup_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    logger.info(f"\nSaved → {out_path}")

    # Print summary table
    _print_summary_table(results)


def _print_summary_table(results):
    models  = list(results.keys())
    subgrps = ["overall", "miss_low", "miss_high", "lab_miss_high",
               "first_6h", "first_12h", "first_24h", "long_stay"]

    print("\n" + "="*90)
    print(f"{'Subgroup':<20}" + "".join(f"{m:<14}" for m in models))
    print("="*90)
    for sg in subgrps:
        row = f"{sg:<20}"
        for m in models:
            r = results.get(m, {}).get(sg)
            if r and r.get("auroc") is not None:
                row += f"{r['auroc']:.4f}        "
            else:
                row += "  N/A          "
        print(row)
    print("="*90)


if __name__ == "__main__":
    main()
