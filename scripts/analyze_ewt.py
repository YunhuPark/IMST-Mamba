"""
B3: Early Warning Time (EWT) analysis — all models vs clinical scores.

EWT = hours between first alarm and sepsis onset.
  Positive EWT = alarm BEFORE onset (early detection — clinically valuable).
  Negative EWT = alarm AFTER onset (late detection — missed opportunity).

Computes:
  1. EWT for all models (IMST-Mamba, GRU-D, LSTM, Transformer, RETAIN)
  2. EWT for clinical scores (qSOFA≥2, SOFA≥2)
  3. Statistical significance (Wilcoxon signed-rank vs qSOFA)
  4. EWT stratified by sepsis severity (SOFA at onset)
  5. Sensitivity at EWT > 0 (any early alarm) and EWT > 3h

Usage:
    python scripts/analyze_ewt.py
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "imst_mamba":  ("src.models.imst_mamba",          "build_model", "configs/model_imst_mamba.yaml"),
    "grud":        ("src.models.grud",                 "build_model", "configs/model_grud.yaml"),
    "lstm":        ("src.models.lstm_baseline",        "build_model", "configs/model_lstm.yaml"),
    "transformer": ("src.models.transformer_baseline", "build_model", "configs/model_transformer.yaml"),
    "retain":      ("src.models.retain",               "build_model", None),
}

SBP_IDX  = 3
RESP_IDX = 6
QSOFA_THRESHOLD  = 2
SOFA_THRESHOLD   = 2


def find_threshold_at_spec(y_true: np.ndarray, probs: np.ndarray,
                            target_spec: float = 0.90) -> float:
    """Return probability threshold giving closest specificity to target."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    spec = 1 - fpr
    idx = np.argmin(np.abs(spec - target_spec))
    return float(thresholds[idx])


# ---------------------------------------------------------------------------
# Clinical score helpers
# ---------------------------------------------------------------------------

def compute_qsofa_series(x_norm: np.ndarray, m: np.ndarray,
                          mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_raw = x_norm * std + mean
    T = x_raw.shape[0]
    sbp_last = resp_last = np.nan
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if m[t, SBP_IDX]  > 0.5: sbp_last  = float(x_raw[t, SBP_IDX])
        if m[t, RESP_IDX] > 0.5: resp_last = float(x_raw[t, RESP_IDX])
        score = 0
        if not np.isnan(sbp_last)  and sbp_last  <= 100.0: score += 1
        if not np.isnan(resp_last) and resp_last >= 22.0:   score += 1
        scores[t] = score
    return scores


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, cfg: dict, save_dir: Path, device: torch.device):
    mod_path, fn_name, cfg_path = MODEL_REGISTRY[model_name]
    ckpt_paths = sorted(save_dir.glob(f"*/checkpoints/{model_name}*_best.pt"))
    if not ckpt_paths:
        return None, []

    if cfg_path and Path(cfg_path).exists():
        model_cfg = load_config("configs/base.yaml", cfg_path)
    else:
        model_cfg = cfg

    mod      = importlib.import_module(mod_path)
    build_fn = getattr(mod, fn_name)
    return build_fn(model_cfg).to(device), ckpt_paths


def infer_model_probs(model, ckpt_paths, test_loader, device) -> dict:
    """Per-patient model probabilities averaged over seeds."""
    seed_preds: dict = {}

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                xb   = batch["x"].to(device)
                mb   = batch["m"].to(device)
                dtb  = batch["delta_t"].to(device)
                sb   = batch["s"].to(device)
                attn = batch["attention_mask"].to(device)
                sids = batch["stay_ids"]
                attn_cpu = batch["attention_mask"]

                out = model(xb, mb, dtb, sb, attn)
                p   = torch.sigmoid(out["logit_sepsis"]).squeeze(-1).cpu()

                for i, sid in enumerate(sids):
                    T = int(attn_cpu[i].sum())
                    if sid not in seed_preds:
                        seed_preds[sid] = []
                    seed_preds[sid].append(p[i, :T].numpy())

    # Average across seeds
    return {sid: np.mean(preds, axis=0) for sid, preds in seed_preds.items()}


# ---------------------------------------------------------------------------
# EWT computation
# ---------------------------------------------------------------------------

def compute_ewt_for_patients(patient_records: dict,
                              pred_key: str,
                              threshold: float) -> dict:
    """
    For each sepsis patient, compute EWT = onset_hour - first_alarm_hour.
    Returns dict: stay_id → {ewt, sofa_at_onset, onset_hour, alarm_hour}
    """
    ewt_dict = {}

    for sid, d in patient_records.items():
        y   = d["labels"]
        dt  = d["delta_t_h"]
        s   = d[pred_key]
        sofa = d.get("sofa", np.full(len(y), np.nan))

        # Find sepsis onset
        valid = ~np.isnan(y)
        cum_h = np.cumsum(dt)

        onset_idx = None
        for t in range(len(y)):
            if valid[t] and y[t] >= 0.5:
                onset_idx = t
                break
        if onset_idx is None:
            continue   # non-sepsis patient

        onset_hour = float(cum_h[onset_idx])
        sofa_onset = float(sofa[onset_idx]) if not np.isnan(sofa[onset_idx]) else float("nan")

        # Find first alarm
        alarm_idx = None
        for t in range(len(s)):
            if s[t] >= threshold:
                alarm_idx = t
                break

        if alarm_idx is None:
            ewt_dict[sid] = {
                "ewt":          float("nan"),   # alarm never raised
                "sofa_onset":   sofa_onset,
                "onset_hour":   onset_hour,
                "alarm_hour":   float("nan"),
                "tp":           False,
            }
        else:
            alarm_hour = float(cum_h[alarm_idx])
            ewt_dict[sid] = {
                "ewt":          onset_hour - alarm_hour,
                "sofa_onset":   sofa_onset,
                "onset_hour":   onset_hour,
                "alarm_hour":   alarm_hour,
                "tp":           (alarm_idx <= onset_idx),
            }

    return ewt_dict


def summarize_ewt(ewt_dict: dict) -> dict:
    ewts     = [v["ewt"] for v in ewt_dict.values() if not np.isnan(v["ewt"])]
    detected = [v for v in ewt_dict.values() if not np.isnan(v["ewt"])]

    n_sepsis  = len(ewt_dict)
    n_alerted = len(detected)
    n_early   = sum(1 for e in ewts if e > 0)
    n_early3h = sum(1 for e in ewts if e > 3)

    return {
        "n_sepsis":          n_sepsis,
        "n_alerted":         n_alerted,
        "detection_rate":    n_alerted / (n_sepsis + 1e-10),
        "pct_early":         n_early / (n_alerted + 1e-10),
        "pct_early_3h":      n_early3h / (n_alerted + 1e-10),
        "mean_ewt_h":        float(np.mean(ewts)) if ewts else float("nan"),
        "median_ewt_h":      float(np.median(ewts)) if ewts else float("nan"),
        "std_ewt_h":         float(np.std(ewts)) if ewts else float("nan"),
        "raw_ewts":          ewts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg      = load_config("configs/base.yaml")
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info(f"Device: {device}")

    with open(processed_dir / "stats.json") as f:
        stats = json.load(f)
    mean_np = np.array(stats["mean"], dtype=np.float32)
    std_np  = np.array(stats["std"],  dtype=np.float32)

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    # ── Collect base patient data (clinical scores) ───────────────────────
    logger.info("Collecting patient-level data and clinical scores...")
    patient_records: dict = {}

    for batch in test_loader:
        x    = batch["x"]
        m    = batch["m"]
        y    = batch["y"]
        sofa = batch["sofa"]
        dt   = batch["delta_t"]      # in hours
        attn = batch["attention_mask"]
        sids = batch["stay_ids"]

        for i, sid in enumerate(sids):
            T = int(attn[i].sum())
            x_i   = x[i, :T].numpy()
            m_i   = m[i, :T].numpy()
            y_i   = y[i, :T].numpy()
            so_i  = sofa[i, :T].numpy()
            dt_i  = dt[i, :T].numpy()

            patient_records[sid] = {
                "labels":    y_i,
                "delta_t_h": dt_i,
                "qsofa":     compute_qsofa_series(x_i, m_i, mean_np, std_np),
                "sofa":      so_i,
            }

    # ── Run inference for all models ──────────────────────────────────────
    all_ewt_results = {}

    for model_name in MODEL_REGISTRY:
        logger.info(f"\n{'='*40}\nModel: {model_name}")
        model, ckpt_paths = load_model(model_name, cfg, save_dir, device)
        if model is None:
            logger.warning(f"  No checkpoint for {model_name}")
            continue

        preds = infer_model_probs(model, ckpt_paths, test_loader, device)
        del model

        for sid in patient_records:
            if sid in preds:
                patient_records[sid][model_name] = preds[sid]
            else:
                patient_records[sid][model_name] = np.zeros(len(patient_records[sid]["labels"]))

        # Compute per-model threshold at 90% specificity
        all_probs, all_labels = [], []
        for d in patient_records.values():
            if model_name not in d:
                continue
            valid = ~np.isnan(d["labels"])
            if valid.sum() == 0:
                continue
            all_probs.append(d[model_name][valid])
            all_labels.append(d["labels"][valid])
        if not all_probs:
            continue
        y_flat = np.concatenate(all_labels)
        p_flat = np.concatenate(all_probs)
        model_thresh = find_threshold_at_spec(y_flat, p_flat, target_spec=0.90)
        logger.info(f"  {model_name} Se@Sp90 threshold: {model_thresh:.4f}")

        ewt_dict = compute_ewt_for_patients(patient_records, model_name, model_thresh)
        summ = summarize_ewt(ewt_dict)
        summ["threshold"] = model_thresh
        all_ewt_results[model_name] = summ

        logger.info(f"  EWT: mean={summ['mean_ewt_h']:.2f}h  "
                    f"median={summ['median_ewt_h']:.2f}h  "
                    f"early%={100*summ['pct_early']:.1f}%  "
                    f"early3h%={100*summ['pct_early_3h']:.1f}%  "
                    f"detected={summ['n_alerted']}/{summ['n_sepsis']}")

    # ── Clinical score EWT ───────────────────────────────────────────────
    for score_key, threshold, label in [
        ("qsofa", QSOFA_THRESHOLD, "qSOFA≥2"),
        ("sofa",  SOFA_THRESHOLD,  "SOFA≥2"),
    ]:
        ewt_dict = compute_ewt_for_patients(patient_records, score_key, threshold)
        summ = summarize_ewt(ewt_dict)
        all_ewt_results[label] = summ
        logger.info(f"{label}: mean={summ['mean_ewt_h']:.2f}h  "
                    f"early%={100*summ['pct_early']:.1f}%  "
                    f"detected={summ['n_alerted']}/{summ['n_sepsis']}")

    # ── Statistical significance (vs qSOFA) ──────────────────────────────
    logger.info("\nWilcoxon signed-rank tests vs qSOFA≥2...")
    qsofa_ewts = all_ewt_results.get("qSOFA≥2", {}).get("raw_ewts", [])
    stat_tests = {}

    for name, res in all_ewt_results.items():
        if name == "qSOFA≥2" or not res.get("raw_ewts"):
            continue
        model_ewts = res["raw_ewts"]
        n = min(len(qsofa_ewts), len(model_ewts))
        if n < 10:
            continue
        try:
            stat, pval = scipy_stats.wilcoxon(model_ewts[:n], qsofa_ewts[:n],
                                               alternative="greater")
            stat_tests[name] = {"statistic": float(stat), "p_value": float(pval)}
            logger.info(f"  {name} vs qSOFA: W={stat:.1f}  p={pval:.4f}")
        except Exception as e:
            logger.warning(f"  {name}: {e}")

    # ── EWT by SOFA severity tier ─────────────────────────────────────────
    logger.info("\nEWT stratified by SOFA at sepsis onset...")
    severity_ewt = {}

    ref_model = "imst_mamba" if "imst_mamba" in all_ewt_results else None
    if ref_model:
        ref_thresh = all_ewt_results[ref_model].get("threshold", 0.5)
        ewt_dict = compute_ewt_for_patients(patient_records, ref_model, ref_thresh)
        for tier, lo, hi in [("mild", 0, 4), ("moderate", 4, 8), ("severe", 8, 100)]:
            tier_ewts = [v["ewt"] for v in ewt_dict.values()
                         if not np.isnan(v["ewt"]) and not np.isnan(v["sofa_onset"])
                         and lo <= v["sofa_onset"] < hi]
            if tier_ewts:
                severity_ewt[tier] = {
                    "mean_ewt_h":   float(np.mean(tier_ewts)),
                    "median_ewt_h": float(np.median(tier_ewts)),
                    "pct_early":    float((np.array(tier_ewts) > 0).mean()),
                    "n":            len(tier_ewts),
                }
                logger.info(f"  SOFA {tier} (SOFA {lo}-{hi}): "
                            f"mean EWT={np.mean(tier_ewts):.2f}h  n={len(tier_ewts)}")

    # ── Save ──────────────────────────────────────────────────────────────
    # Remove raw_ewts from save (too large) — keep summary stats only
    save_results = {}
    for k, v in all_ewt_results.items():
        d = dict(v)
        d.pop("raw_ewts", None)
        save_results[k] = d

    out = {
        "ewt_summary":        save_results,
        "statistical_tests":  stat_tests,
        "severity_stratified": severity_ewt,
        "threshold_model":    MODEL_THRESHOLD,
        "threshold_qsofa":    QSOFA_THRESHOLD,
        "threshold_sofa":     SOFA_THRESHOLD,
    }
    out_path = save_dir / "ewt_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    logger.info(f"\nSaved → {out_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Method':<20} {'Mean EWT(h)':>12} {'Median':>8} "
          f"{'%Early':>8} {'%Early>3h':>10} {'DetRate':>8}")
    print("="*80)
    for name in ["IMST-Mamba", "imst_mamba", "grud", "lstm", "transformer", "retain",
                 "qSOFA≥2", "SOFA≥2"]:
        r = all_ewt_results.get(name) or all_ewt_results.get(name.lower())
        if not r:
            continue
        label = name if name in ["qSOFA≥2", "SOFA≥2"] else name
        print(f"{label:<20} {r['mean_ewt_h']:>12.2f} {r['median_ewt_h']:>8.2f} "
              f"{100*r['pct_early']:>7.1f}% {100*r['pct_early_3h']:>9.1f}% "
              f"{100*r['detection_rate']:>7.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
