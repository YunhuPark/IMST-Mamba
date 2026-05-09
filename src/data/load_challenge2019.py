"""
Load PhysioNet/CinC Challenge 2019 data into model-ready tensors.

Data source: physionet.org/content/challenge-2019/1.0.0/
  data/raw/training_setA/  (~20k PSV files)
  data/raw/training_setB/  (~20k PSV files)

Each PSV file = one ICU patient, hourly rows.
Columns: HR|O2Sat|...|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel

Pipeline (replaces MIMIC extract_cohort + extract_features +
          generate_labels + build_timeseries):

  load_challenge2019()  →  data/processed/raw/stay_{pid}.pt
                           data/interim/cohort.parquet
                           data/interim/labels.parquet
                           data/processed/valid_stay_ids.json

  normalize_and_copy()  →  data/processed/{train,val,test}/stay_{pid}.pt
                           data/processed/stats.json

Tensor contents per patient:
  x         (T, 34)  feature values (0 where unobserved)
  m         (T, 34)  observation mask (1 where observed)
  delta_t   (T,)     seconds since previous hour (always 3600 except t=0)
  s         (T, 34)  seconds since last observation of each feature (1e9 if never)
  miss_state(T, 34)  3-state missingness {0=never, 1=recent, 2=stale}
  y_6h      (T,)     label: 1 if sepsis within next 6h, 0 otherwise, NaN if already septic
  y_12h     (T,)     same for 12h horizon
  y_24h     (T,)     same for 24h horizon
  sofa      (T,)     zeros (SOFA not directly available in Challenge data)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.challenge_utils import (
    FEATURE_NAMES, N_FEATURES, CLIP_RANGES, get_recency_thresholds_array
)

logger = logging.getLogger(__name__)

# Seconds per hour (Challenge data is already hourly)
HOUR_SECS = 3600.0

# Prediction horizons (hours)
HORIZONS = [6, 12, 24]


def _psv_patient_id(path: Path) -> int:
    """Convert filename p000001.psv → integer 1, p100001.psv → 100001."""
    return int(path.stem[1:])


def _process_patient(
    psv_path: Path,
    min_stay_hours: int,
    tau: np.ndarray,
) -> dict | None:
    """
    Process a single patient PSV file into model tensors.

    Returns None if the patient is filtered out (too short).
    """
    try:
        df = pd.read_csv(psv_path, sep="|")
    except Exception as e:
        logger.warning(f"Failed to read {psv_path}: {e}")
        return None

    T = len(df)
    if T < min_stay_hours:
        return None

    # ── Features ──────────────────────────────────────────────────────────────
    feat_df = df[FEATURE_NAMES]
    m = (~feat_df.isna()).values.astype(np.float32)   # (T, F)
    x = feat_df.fillna(0.0).values.astype(np.float32)  # (T, F)

    # Clip outliers (only where observed)
    for fi, fname in enumerate(FEATURE_NAMES):
        lo, hi = CLIP_RANGES[fname]
        x[:, fi] = np.clip(x[:, fi], lo, hi) * m[:, fi]

    # ── delta_t (seconds since previous time step) ────────────────────────────
    # Challenge data is hourly → always 3600s between steps, 0 for first step
    delta_t = np.full(T, HOUR_SECS, dtype=np.float32)
    delta_t[0] = 0.0

    # ── s (seconds since last observation of each feature) ────────────────────
    s = np.full((T, N_FEATURES), 1e9, dtype=np.float32)
    t_arr = np.arange(T, dtype=np.float32)
    for fi in range(N_FEATURES):
        obs_times = np.where(m[:, fi] > 0)[0]
        if len(obs_times) == 0:
            continue  # never observed — s stays at 1e9
        # For t >= obs_times[0]: most recent obs = obs_times[searchsorted-1]
        t_from = int(obs_times[0])
        idx = np.searchsorted(obs_times, np.arange(t_from, T), side="right") - 1
        last_obs = obs_times[idx].astype(np.float32)
        s[t_from:, fi] = (t_arr[t_from:] - last_obs) * HOUR_SECS

    # ── miss_state (3-state per feature) ──────────────────────────────────────
    # 0 = never observed, 1 = recently observed (s <= τ), 2 = stale (s > τ)
    miss_state = np.zeros((T, N_FEATURES), dtype=np.int64)
    ever_seen = np.zeros(N_FEATURES, dtype=bool)
    for t in range(T):
        ever_seen |= (m[t] > 0)
        stale = (s[t] > tau) & ever_seen
        recent = (s[t] <= tau) & ever_seen
        miss_state[t, stale] = 2
        miss_state[t, recent] = 1
        # miss_state[t, ~ever_seen] stays 0

    # ── Labels ────────────────────────────────────────────────────────────────
    sepsis_arr = df["SepsisLabel"].fillna(0).astype(int).values  # (T,)
    onset_arr = np.where(sepsis_arr == 1)[0]
    sepsis_onset = int(onset_arr[0]) if len(onset_arr) > 0 else -1

    y_labels: dict[str, np.ndarray] = {}
    for h in HORIZONS:
        y = np.zeros(T, dtype=np.float32)
        if sepsis_onset >= 0:
            # Rows at/after onset: NaN (already septic, not a prediction target)
            y[sepsis_onset:] = float("nan")
            # Rows within h hours before onset: positive
            pos_start = max(0, sepsis_onset - h)
            y[pos_start:sepsis_onset] = 1.0
        y_labels[f"y_{h}h"] = y

    # ── Pack tensors ──────────────────────────────────────────────────────────
    return {
        "x":          torch.from_numpy(x),
        "m":          torch.from_numpy(m),
        "delta_t":    torch.from_numpy(delta_t),
        "s":          torch.from_numpy(s),
        "miss_state": torch.from_numpy(miss_state),
        "y_6h":       torch.from_numpy(y_labels["y_6h"]),
        "y_12h":      torch.from_numpy(y_labels["y_12h"]),
        "y_24h":      torch.from_numpy(y_labels["y_24h"]),
        "sofa":       torch.zeros(T, dtype=torch.float32),
    }


def _read_patient_meta(psv_path: Path, pid: int) -> dict:
    """Read demographic metadata from a PSV file (first row only)."""
    try:
        df = pd.read_csv(psv_path, sep="|", nrows=1)
        age = float(df["Age"].iloc[0]) if not pd.isna(df["Age"].iloc[0]) else 50.0
        unit1 = int(df["Unit1"].iloc[0]) if not pd.isna(df["Unit1"].iloc[0]) else 0
        unit2 = int(df["Unit2"].iloc[0]) if not pd.isna(df["Unit2"].iloc[0]) else 0
        los = len(pd.read_csv(psv_path, sep="|"))
    except Exception:
        age, unit1, unit2, los = 50.0, 0, 0, 0

    # Map Unit1/Unit2 flags to ICU type string
    if unit1 == 1:
        icu_type = "MICU"
    elif unit2 == 1:
        icu_type = "SICU"
    else:
        icu_type = "Other"

    return {
        "subject_id":    pid,
        "stay_id":       pid,
        "age":           age,
        "first_careunit": icu_type,
        "los_hours":     los,
    }


def _has_sepsis(psv_path: Path) -> bool:
    """Return True if this patient develops sepsis (any SepsisLabel==1)."""
    try:
        df = pd.read_csv(psv_path, sep="|", usecols=["SepsisLabel"])
        return bool((df["SepsisLabel"].fillna(0) == 1).any())
    except Exception:
        return False


def load_challenge2019(
    cfg: dict,
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
) -> None:
    """
    Main entry point. Reads all PSV files and produces:
      - data/processed/raw/stay_{pid}.pt  (per patient tensors)
      - data/interim/cohort.parquet
      - data/interim/labels.parquet
      - data/processed/valid_stay_ids.json
    """
    # Find all PSV files
    psv_files: list[Path] = []
    for subset in ["training_setA", "training_setB"]:
        subset_dir = raw_dir / subset
        if subset_dir.exists():
            psv_files.extend(sorted(subset_dir.glob("*.psv")))
    if not psv_files:
        # Flat layout fallback
        psv_files = sorted(raw_dir.glob("*.psv"))

    if not psv_files:
        raise FileNotFoundError(
            f"No PSV files found in {raw_dir}. "
            "Download training_setA and training_setB from "
            "physionet.org/content/challenge-2019/1.0.0/"
        )

    logger.info(f"Found {len(psv_files):,} PSV files")

    min_stay_hours = cfg["data"]["min_stay_hours"]
    tau = get_recency_thresholds_array()  # (F,) in seconds

    raw_out = processed_dir / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    cohort_rows: list[dict] = []
    label_rows: list[dict] = []
    valid_ids: list[int] = []

    for psv_path in tqdm(psv_files, desc="Processing patients", unit="pt"):
        pid = _psv_patient_id(psv_path)
        tensors = _process_patient(psv_path, min_stay_hours, tau)
        if tensors is None:
            continue

        # Save tensor
        out_path = raw_out / f"stay_{pid}.pt"
        torch.save(tensors, out_path)

        valid_ids.append(pid)

        # Metadata for cohort/labels
        meta = _read_patient_meta(psv_path, pid)
        cohort_rows.append(meta)

        has_sep = bool((tensors["y_6h"].nan_to_num(-1) == float("nan")).any() or
                       any(tensors[f"y_{h}h"].isnan().any() for h in HORIZONS))
        # Simpler: check if any NaN exists in y_6h (NaN means post-onset)
        has_sep = bool(tensors["y_6h"].isnan().any())
        label_rows.append({
            "stay_id":   pid,
            "has_sepsis": has_sep,
        })

    logger.info(
        f"Valid patients: {len(valid_ids):,} / {len(psv_files):,}  "
        f"(filtered {len(psv_files) - len(valid_ids):,} short stays)"
    )
    logger.info(
        f"Sepsis prevalence: "
        f"{sum(r['has_sepsis'] for r in label_rows):,} / {len(label_rows):,}"
    )

    # Save cohort and labels parquets (used by splits.py)
    cohort_df = pd.DataFrame(cohort_rows)
    labels_df = pd.DataFrame(label_rows)
    cohort_df.to_parquet(interim_dir / "cohort.parquet", index=False)
    labels_df.to_parquet(interim_dir / "labels.parquet", index=False)

    # Save valid IDs
    with open(processed_dir / "valid_stay_ids.json", "w") as f:
        json.dump(valid_ids, f)

    logger.info(f"Cohort saved → {interim_dir / 'cohort.parquet'}")
    logger.info(f"Labels saved → {interim_dir / 'labels.parquet'}")
    logger.info(f"Valid IDs saved → {processed_dir / 'valid_stay_ids.json'}")


def normalize_and_copy(processed_dir: Path, splits: dict) -> None:
    """
    Compute normalization statistics from the training split,
    then normalize and copy all tensors to train/val/test subdirectories.

    Stats are saved to data/processed/stats.json.
    """
    raw_dir = processed_dir / "raw"

    logger.info("Computing normalization statistics from training split...")
    sum_x = np.zeros(N_FEATURES, dtype=np.float64)
    sum_sq = np.zeros(N_FEATURES, dtype=np.float64)
    count = np.zeros(N_FEATURES, dtype=np.float64)

    for sid in tqdm(splits["train"], desc="Computing stats", unit="pt"):
        pt_path = raw_dir / f"stay_{sid}.pt"
        if not pt_path.exists():
            continue
        data = torch.load(pt_path, weights_only=True)
        x = data["x"].numpy()   # (T, F)
        m = data["m"].numpy()   # (T, F)
        # Accumulate only observed values
        for fi in range(N_FEATURES):
            obs = x[m[:, fi] > 0.5, fi]
            if len(obs) > 0:
                sum_x[fi] += obs.sum()
                sum_sq[fi] += (obs ** 2).sum()
                count[fi] += len(obs)

    mean = np.where(count > 0, sum_x / count, 0.0).astype(np.float32)
    var = np.where(count > 0, sum_sq / count - (sum_x / count) ** 2, 1.0)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    stats = {
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved → {processed_dir / 'stats.json'}")

    # Normalize and copy to split directories
    for split_name, stay_ids in splits.items():
        split_dir = processed_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Copying {split_name} ({len(stay_ids):,} patients)...")
        for sid in tqdm(stay_ids, desc=f"Normalizing {split_name}", unit="pt"):
            src = raw_dir / f"stay_{sid}.pt"
            dst = split_dir / f"stay_{sid}.pt"
            if not src.exists():
                continue

            data = torch.load(src, weights_only=True)
            x = data["x"].numpy().copy()    # (T, F)
            m = data["m"].numpy()

            # Normalize observed values; leave unobserved (m=0) as 0
            x_norm = (x - mean) / std
            x_norm = x_norm * m  # zero out unobserved positions

            data["x"] = torch.from_numpy(x_norm.astype(np.float32))
            torch.save(data, dst)

    logger.info("Normalize and copy complete.")
