"""
B1: Clinical score comparison — qSOFA and modified SOFA vs IMST-Mamba.

Computes per-timestep scores and then evaluates:
  1. qSOFA (2-criteria: SBP≤100, Resp≥22 — GCS unavailable)
  2. Modified SOFA from features (5 components: respiratory, coagulation,
     liver, cardiovascular, renal — CNS/vasopressors unavailable)
  3. IMST-Mamba predictions (averaged across seeds)

Metrics: AUROC, AUPRC, Sensitivity, Specificity at clinical thresholds
Early detection: mean hours before sepsis onset for each method

Usage:
    python scripts/analyze_clinical_comparison.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import full_metrics, print_metrics
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Feature indices (PhysioNet 2019 34-feature set)
SBP_IDX   = 3    # Systolic Blood Pressure (mmHg)
RESP_IDX  = 6    # Respiratory Rate (breaths/min)
FIO2_IDX  = 10   # FiO2 (fraction, 0.21–1.0)
SAO2_IDX  = 13   # SaO2 (%)
CREAT_IDX = 19   # Creatinine (mg/dL)
BILI_IDX  = 26   # Bilirubin_total (mg/dL)
MAP_IDX   = 4    # Mean Arterial Pressure (mmHg)
PLT_IDX   = 33   # Platelets (10^9/L)


# ---------------------------------------------------------------------------
# Clinical score computation
# ---------------------------------------------------------------------------

def _sofa_respiratory(sao2, fio2):
    if np.isnan(sao2) or np.isnan(fio2) or fio2 <= 0:
        return 0
    ratio = sao2 / fio2
    if ratio >= 400: return 0
    if ratio >= 300: return 1
    if ratio >= 200: return 2
    return 3

def _sofa_coagulation(plt):
    if np.isnan(plt): return 0
    if plt >= 150: return 0
    if plt >= 100: return 1
    if plt >= 50:  return 2
    if plt >= 20:  return 3
    return 4

def _sofa_liver(bili):
    if np.isnan(bili): return 0
    if bili < 1.2:  return 0
    if bili < 2.0:  return 1
    if bili < 6.0:  return 2
    if bili < 12.0: return 3
    return 4

def _sofa_cardiovascular(map_val):
    if np.isnan(map_val): return 0
    return 0 if map_val >= 70 else 1

def _sofa_renal(creat):
    if np.isnan(creat): return 0
    if creat < 1.2:  return 0
    if creat < 2.0:  return 1
    if creat < 3.5:  return 2
    if creat < 5.0:  return 3
    return 4


def compute_qsofa_series(x_norm, m, mean, std):
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


def compute_modified_sofa_series(x_norm, m, mean, std):
    """
    5-component modified SOFA from available features (max score 16).
    Components: respiratory, coagulation, liver, cardiovascular, renal.
    GCS (CNS) and vasopressors unavailable in PhysioNet 2019 features.
    Uses carry-forward for missing observations.
    """
    x_raw = x_norm * std + mean
    T = x_raw.shape[0]

    feats = {
        'sao2': SAO2_IDX, 'fio2': FIO2_IDX, 'creat': CREAT_IDX,
        'bili':  BILI_IDX, 'map':  MAP_IDX,  'plt':   PLT_IDX,
    }
    last = {k: np.nan for k in feats}
    scores = np.zeros(T, dtype=np.float32)

    for t in range(T):
        for k, idx in feats.items():
            if m[t, idx] > 0.5:
                last[k] = float(x_raw[t, idx])

        score = (
            _sofa_respiratory(last['sao2'], last['fio2']) +
            _sofa_coagulation(last['plt']) +
            _sofa_liver(last['bili']) +
            _sofa_cardiovascular(last['map']) +
            _sofa_renal(last['creat'])
        )
        scores[t] = score
    return scores


def _threshold_metrics(y_true, score, binary_pred):
    try:
        auroc = float(roc_auc_score(y_true, score))
        auprc = float(average_precision_score(y_true, score))
    except Exception:
        auroc = auprc = float("nan")
    tp = float(((binary_pred == 1) & (y_true == 1)).sum())
    tn = float(((binary_pred == 0) & (y_true == 0)).sum())
    fp = float(((binary_pred == 1) & (y_true == 0)).sum())
    fn = float(((binary_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn + 1e-10)
    spec = tn / (tn + fp + 1e-10)
    ppv  = tp / (tp + fp + 1e-10)
    return {"auroc": auroc, "auprc": auprc,
            "sensitivity": sens, "specificity": spec, "ppv": ppv}


def _find_threshold_at_spec(y_true, probs, target_spec=0.90):
    """Return probability threshold giving closest specificity to target."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    spec = 1 - fpr
    idx = np.argmin(np.abs(spec - target_spec))
    return float(thresholds[idx])


def _early_detection_hours(patient_records, threshold, score_key):
    ewts = []
    for d in patient_records.values():
        y   = d["labels"]
        valid = ~np.isnan(y)
        cum  = np.cumsum(d["delta_t_h"])
        onset_idx = next((t for t in range(len(y)) if valid[t] and y[t] >= 0.5), None)
        if onset_idx is None:
            continue
        onset_h = float(cum[onset_idx])
        alarm_idx = next((t for t in range(len(d[score_key])) if d[score_key][t] >= threshold), None)
        if alarm_idx is None:
            continue
        ewts.append(onset_h - float(cum[alarm_idx]))
    return ewts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg       = load_config("configs/base.yaml")
    model_cfg = load_config("configs/base.yaml", "configs/model_imst_mamba.yaml")
    save_dir  = Path(cfg.get("logging", {}).get("save_dir", "results"))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(cfg["data"]["processed_dir"])

    logger.info(f"Device: {device}")

    with open(processed_dir / "stats.json") as f:
        stats = json.load(f)
    mean_np = np.array(stats["mean"], dtype=np.float32)
    std_np  = np.array(stats["std"],  dtype=np.float32)

    _, _, test_loader = build_dataloaders(processed_dir, cfg, horizon="6h")

    # ── Pass 1: clinical scores ───────────────────────────────────────────
    logger.info("Computing clinical scores (qSOFA, modified SOFA)...")
    patient_records = {}

    for batch in test_loader:
        x    = batch["x"]
        m    = batch["m"]
        y    = batch["y"]
        dt   = batch["delta_t"]
        attn = batch["attention_mask"]
        sids = batch["stay_ids"]

        for i, sid in enumerate(sids):
            T = int(attn[i].sum())
            x_i  = x[i, :T].numpy()
            m_i  = m[i, :T].numpy()
            patient_records[sid] = {
                "labels":    y[i, :T].numpy(),
                "delta_t_h": dt[i, :T].numpy(),
                "qsofa":     compute_qsofa_series(x_i, m_i, mean_np, std_np),
                "mod_sofa":  compute_modified_sofa_series(x_i, m_i, mean_np, std_np),
            }

    # ── Pass 2: IMST-Mamba inference ──────────────────────────────────────
    logger.info("Running IMST-Mamba inference...")
    ckpt_paths = sorted(save_dir.glob("*/checkpoints/imst_mamba*_best.pt"))

    if ckpt_paths:
        from src.models.imst_mamba import build_model
        seed_preds = {sid: [] for sid in patient_records}

        for ckpt_path in ckpt_paths:
            model = build_model(model_cfg).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
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
                        seed_preds[sid].append(p[i, :T].numpy())
            del model

        for sid in patient_records:
            if seed_preds[sid]:
                patient_records[sid]["model"] = np.mean(seed_preds[sid], axis=0)
    else:
        logger.warning("No checkpoint found.")
        for sid in patient_records:
            patient_records[sid]["model"] = np.zeros(len(patient_records[sid]["labels"]))

    # ── Flatten for timestep-level metrics ───────────────────────────────
    all_y, all_qsofa, all_sofa, all_model = [], [], [], []
    for d in patient_records.values():
        valid = ~np.isnan(d["labels"])
        if valid.sum() == 0: continue
        all_y.append(d["labels"][valid])
        all_qsofa.append(d["qsofa"][valid])
        all_sofa.append(d["mod_sofa"][valid])
        all_model.append(d["model"][valid])

    y_true     = np.concatenate(all_y)
    qsofa_flat = np.concatenate(all_qsofa)
    sofa_flat  = np.concatenate(all_sofa)
    model_flat = np.concatenate(all_model)

    logger.info(f"Timesteps: {len(y_true):,}  Prevalence: {y_true.mean():.4f}")
    logger.info(f"Modified SOFA range: {sofa_flat.min():.0f}–{sofa_flat.max():.0f}  "
                f"mean={sofa_flat.mean():.2f}")

    # ── Metrics ───────────────────────────────────────────────────────────
    results = {}

    for name, thresh in [("qSOFA≥1", 1), ("qSOFA≥2", 2)]:
        r = _threshold_metrics(y_true, qsofa_flat, (qsofa_flat >= thresh).astype(float))
        results[name] = r
        logger.info(f"{name}: AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
                    f"Sens={r['sensitivity']:.4f}  Spec={r['specificity']:.4f}")

    for name, thresh in [("modSOFA≥2", 2), ("modSOFA≥4", 4), ("modSOFA≥6", 6)]:
        r = _threshold_metrics(y_true, sofa_flat, (sofa_flat >= thresh).astype(float))
        results[name] = r
        logger.info(f"{name}: AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
                    f"Sens={r['sensitivity']:.4f}  Spec={r['specificity']:.4f}")

    if len(np.unique(y_true[~np.isnan(model_flat)])) == 2:
        r = full_metrics(y_true, model_flat, n_bootstrap=200)
        results["IMST-Mamba"] = r
        print_metrics(r, "IMST-Mamba")

    # ── EWT with Se@Sp90 threshold for model ─────────────────────────────
    logger.info("Computing early detection time (EWT)...")
    model_thresh = _find_threshold_at_spec(y_true, model_flat, target_spec=0.90)
    logger.info(f"IMST-Mamba Se@Sp90 threshold: {model_thresh:.4f}")

    ewt_results = {}
    for score_key, thresh, label in [
        ("qsofa",    2.0,          "qSOFA≥2"),
        ("mod_sofa", 2.0,          "modSOFA≥2"),
        ("mod_sofa", 4.0,          "modSOFA≥4"),
        ("model",    model_thresh, "IMST-Mamba@Sp90"),
        ("model",    0.3,          "IMST-Mamba@0.3"),
    ]:
        ewts = _early_detection_hours(patient_records, thresh, score_key)
        if ewts:
            ewt_results[label] = {
                "mean_hours":   float(np.mean(ewts)),
                "median_hours": float(np.median(ewts)),
                "pct_early":    float((np.array(ewts) > 0).mean()),
                "n_patients":   len(ewts),
                "threshold":    thresh,
            }
            logger.info(f"EWT {label}: mean={np.mean(ewts):.2f}h  "
                        f"median={np.median(ewts):.2f}h  "
                        f"early%={100*(np.array(ewts)>0).mean():.1f}%  "
                        f"n={len(ewts)}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {"clinical_metrics": results, "early_detection": ewt_results,
           "model_threshold_sp90": float(model_thresh)}
    out_path = save_dir / "clinical_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    logger.info(f"Saved → {out_path}")

    print("\n" + "="*75)
    print(f"{'Method':<22} {'AUROC':>8} {'AUPRC':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8}")
    print("="*75)
    for name, r in results.items():
        if r and "auroc" in r:
            print(f"{name:<22} {r['auroc']:>8.4f} {r['auprc']:>8.4f} "
                  f"{r.get('sensitivity', float('nan')):>8.4f} "
                  f"{r.get('specificity', float('nan')):>8.4f} "
                  f"{r.get('ppv', float('nan')):>8.4f}")
    print("="*75)

    print("\n" + "="*65)
    print(f"{'Method':<22} {'Mean EWT(h)':>12} {'Median':>8} {'%Early':>8}")
    print("="*65)
    for name, e in ewt_results.items():
        print(f"{name:<22} {e['mean_hours']:>12.2f} {e['median_hours']:>8.2f} "
              f"{100*e['pct_early']:>7.1f}%")
    print("="*65)


if __name__ == "__main__":
    main()
