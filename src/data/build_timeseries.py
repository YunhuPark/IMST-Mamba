"""
Step 4: Build irregular time series tensors from MIMIC-IV features + labels.

For each ICU stay, produces:
  x         : (T, F) float32  — feature values (0 where unobserved)
  m         : (T, F) float32  — binary observation mask
  delta_t   : (T,)   float32  — seconds since previous observation event
  s         : (T, F) float32  — seconds since last obs of each feature (large if never)
  miss_state: (T, F) int8     — 0=never, 1=recent, 2=stale
  y_6h      : (T,)   float32  — sepsis label (6h horizon)
  y_12h     : (T,)   float32  — sepsis label (12h horizon)
  y_24h     : (T,)   float32  — sepsis label (24h horizon)
  sofa      : (T,)   float32  — SOFA total score
  seq_len   : int             — actual number of observation events

Observation event = any time step where >= 1 feature is observed.

Output: data/processed/{split}/stay_{stay_id}.pt  per stay
        data/processed/stats.json                  (mean/std for normalization)
        data/processed/splits.json                 (patient-level splits)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.mimic_utils import FEATURE_NAMES, N_FEATURES, get_recency_thresholds_array

logger = logging.getLogger(__name__)

NEVER_SEEN = 1e9   # sentinel for "feature never observed"


def _build_patient_tensors(
    stay_id: int,
    feat_long: pd.DataFrame,    # long-format: charttime, feature, value
    label_df: pd.DataFrame,     # hourly: charttime, sofa_total, label_6h, label_12h, label_24h
    intime: pd.Timestamp,
    obs_window_h: int,
    tau: np.ndarray,            # recency thresholds (F,)
) -> dict:
    """
    Convert one ICU stay's raw features into tensors.
    """
    # Truncate to observation window
    end_time = intime + pd.Timedelta(hours=obs_window_h)
    feat_long = feat_long[
        (feat_long["charttime"] >= intime) &
        (feat_long["charttime"] <= end_time)
    ].copy()
    label_df = label_df[
        (label_df["charttime"] >= intime) &
        (label_df["charttime"] <= end_time)
    ].copy()

    if len(feat_long) == 0:
        return None

    # Get all unique observation times (sorted)
    obs_times = sorted(feat_long["charttime"].unique())
    T = len(obs_times)

    x = np.zeros((T, N_FEATURES), dtype=np.float32)
    m = np.zeros((T, N_FEATURES), dtype=np.float32)

    # Fill in observed values
    feat_idx_map = {f: i for i, f in enumerate(FEATURE_NAMES)}
    for _, row in feat_long.iterrows():
        t_idx = obs_times.index(row["charttime"])
        f_idx = feat_idx_map.get(row["feature"])
        if f_idx is not None:
            x[t_idx, f_idx] = row["value"]
            m[t_idx, f_idx] = 1.0

    # Compute delta_t (seconds from previous observation event)
    times_sec = np.array(
        [(t - obs_times[0]).total_seconds() for t in obs_times], dtype=np.float32
    )
    delta_t = np.zeros(T, dtype=np.float32)
    delta_t[1:] = np.diff(times_sec)

    # Compute s (time since last observation per feature, seconds)
    s = np.full((T, N_FEATURES), NEVER_SEEN, dtype=np.float32)
    last_obs_time = np.full(N_FEATURES, -NEVER_SEEN, dtype=np.float64)

    for t in range(T):
        t_sec = times_sec[t]
        # Update s BEFORE processing current event
        for f in range(N_FEATURES):
            if last_obs_time[f] > -NEVER_SEEN:
                s[t, f] = float(t_sec - last_obs_time[f])
            # else: remains NEVER_SEEN
        # Update last_obs_time for features observed at this step
        for f in range(N_FEATURES):
            if m[t, f] > 0.5:
                last_obs_time[f] = t_sec

    # Compute 3-state missingness
    # State 0: never seen (s == NEVER_SEEN)
    # State 1: recently seen (s <= tau_f)
    # State 2: stale (seen before but s > tau_f)
    miss_state = np.zeros((T, N_FEATURES), dtype=np.int8)
    for f in range(N_FEATURES):
        never_mask = s[:, f] >= NEVER_SEEN * 0.9
        recent_mask = (~never_mask) & (s[:, f] <= tau[f])
        stale_mask = (~never_mask) & (s[:, f] > tau[f])
        miss_state[never_mask, f] = 0
        miss_state[recent_mask, f] = 1
        miss_state[stale_mask, f] = 2

    # Align labels with observation times
    label_df = label_df.set_index("charttime")
    # Forward-fill labels to obs_times
    label_full = label_df.reindex(
        pd.DatetimeIndex(obs_times), method="ffill"
    ).fillna(0)

    y_6h = label_full["label_6h"].values.astype(np.float32)
    y_12h = label_full["label_12h"].values.astype(np.float32) if "label_12h" in label_full else np.zeros(T, np.float32)
    y_24h = label_full["label_24h"].values.astype(np.float32) if "label_24h" in label_full else np.zeros(T, np.float32)
    sofa = label_full["sofa_total"].values.astype(np.float32) if "sofa_total" in label_full else np.full(T, np.nan, np.float32)

    return {
        "stay_id": stay_id,
        "x": torch.from_numpy(x),
        "m": torch.from_numpy(m),
        "delta_t": torch.from_numpy(delta_t),
        "s": torch.from_numpy(s),
        "miss_state": torch.from_numpy(miss_state).long(),
        "y_6h": torch.from_numpy(y_6h),
        "y_12h": torch.from_numpy(y_12h),
        "y_24h": torch.from_numpy(y_24h),
        "sofa": torch.from_numpy(sofa),
        "seq_len": T,
    }




def build_timeseries(cfg: dict, interim_dir: Path, processed_dir: Path,
                     splits: Optional[dict] = None) -> None:
    """
    Build all patient tensors. Normalization is applied after splits are known.
    """
    obs_window = cfg["data"]["observation_window"]
    tau = get_recency_thresholds_array()

    cohort = pd.read_parquet(interim_dir / "cohort.parquet")
    features = pd.read_parquet(interim_dir / "features.parquet")
    labels = pd.read_parquet(interim_dir / "labels.parquet")

    logger.info(f"Building tensors for {len(cohort):,} stays...")

    # First pass: build raw (un-normalized) tensors
    raw_dir = processed_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    valid_stay_ids = []
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Tensors"):
        sid = row["stay_id"]
        feat_stay = features[features["stay_id"] == sid]
        label_stay = labels[labels["stay_id"] == sid]

        tensors = _build_patient_tensors(
            sid, feat_stay, label_stay,
            row["intime"], obs_window, tau
        )
        if tensors is None or tensors["seq_len"] < 4:
            continue

        torch.save(tensors, raw_dir / f"stay_{sid}.pt")
        valid_stay_ids.append(sid)

    logger.info(f"Valid stays: {len(valid_stay_ids):,}")

    # Save valid stay list for splits.py to use
    with open(processed_dir / "valid_stay_ids.json", "w") as f:
        json.dump(valid_stay_ids, f)

    logger.info("Done building tensors. Run splits.py next, then normalize_and_split.py")


def normalize_and_copy(processed_dir: Path, splits_dict: dict) -> None:
    """
    Compute normalization stats from train split, apply to all splits,
    and save to processed_dir/{train,val,test}/.
    """
    logger.info("Computing normalization stats from train split...")
    raw_dir = processed_dir / "raw"

    # Collect observed values per feature from training set
    sum_vals = np.zeros(N_FEATURES, dtype=np.float64)
    sum_sq = np.zeros(N_FEATURES, dtype=np.float64)
    counts = np.zeros(N_FEATURES, dtype=np.float64)

    for sid in tqdm(splits_dict["train"], desc="Stats"):
        p = raw_dir / f"stay_{sid}.pt"
        if not p.exists():
            continue
        data = torch.load(p, weights_only=True)
        x = data["x"].numpy()   # (T, F)
        m = data["m"].numpy()   # (T, F)
        for f in range(N_FEATURES):
            obs = x[m[:, f] > 0.5, f]
            if len(obs) > 0:
                sum_vals[f] += obs.sum()
                sum_sq[f] += (obs ** 2).sum()
                counts[f] += len(obs)

    mean = np.where(counts > 0, sum_vals / counts, 0.0)
    var = np.where(counts > 0, sum_sq / counts - mean ** 2, 1.0)
    std = np.sqrt(np.maximum(var, 1e-6))

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved normalization stats → {processed_dir / 'stats.json'}")

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    # Copy normalized tensors to split directories
    for split_name, stay_ids in splits_dict.items():
        out_dir = processed_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid in tqdm(stay_ids, desc=f"Normalizing {split_name}"):
            p = raw_dir / f"stay_{sid}.pt"
            if not p.exists():
                continue
            data = torch.load(p, weights_only=True)
            x = data["x"]
            m = data["m"]
            # Normalize observed values only (unobserved stay 0)
            x_norm = (x - mean_t) / std_t
            x_norm = x_norm * m   # zero out unobserved positions
            data["x"] = x_norm
            data["x_mean"] = mean_t
            data["x_std"] = std_t
            torch.save(data, out_dir / f"stay_{sid}.pt")

    logger.info("Normalization complete.")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config("configs/base.yaml")
    build_timeseries(
        cfg,
        interim_dir=Path(cfg["data"]["interim_dir"]),
        processed_dir=Path(cfg["data"]["processed_dir"]),
    )


if __name__ == "__main__":
    main()
