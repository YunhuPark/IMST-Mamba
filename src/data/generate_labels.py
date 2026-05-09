"""
Step 3: Generate Sepsis-3 labels using SOFA score + infection window.

Sepsis-3 definition (Singer et al., JAMA 2016):
  - Suspected infection: antibiotic + blood culture within 72h window
  - Acute organ dysfunction: SOFA total increase >= 2 from baseline

Label: y(t) = 1 if sepsis onset occurs within prediction_horizon hours of time t.

Output: data/interim/labels.parquet
Columns: stay_id, charttime, sofa_total, sofa_resp, sofa_coag, sofa_liver,
         sofa_cardio, sofa_neuro, sofa_renal, label_6h, label_12h, label_24h,
         sepsis_onset_time, has_sepsis
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.mimic_utils import is_antibiotic

logger = logging.getLogger(__name__)


# ── SOFA scoring functions ────────────────────────────────────────────────────

def _sofa_respiratory(pao2: float, fio2: float, on_vent: bool = False) -> int:
    """PaO2/FiO2 ratio → SOFA respiratory score."""
    if np.isnan(pao2) or np.isnan(fio2) or fio2 <= 0:
        return np.nan
    ratio = pao2 / fio2
    if ratio >= 400:
        return 0
    elif ratio >= 300:
        return 1
    elif ratio >= 200:
        return 2
    elif ratio >= 100:
        return 3
    else:
        return 4


def _sofa_coagulation(platelets: float) -> int:
    """Platelets (×10³/μL) → SOFA coagulation score."""
    if np.isnan(platelets):
        return np.nan
    if platelets >= 150:
        return 0
    elif platelets >= 100:
        return 1
    elif platelets >= 50:
        return 2
    elif platelets >= 20:
        return 3
    else:
        return 4


def _sofa_liver(bilirubin: float) -> int:
    """Total bilirubin (mg/dL) → SOFA liver score."""
    if np.isnan(bilirubin):
        return np.nan
    if bilirubin < 1.2:
        return 0
    elif bilirubin < 2.0:
        return 1
    elif bilirubin < 6.0:
        return 2
    elif bilirubin < 12.0:
        return 3
    else:
        return 4


def _sofa_cardiovascular(map_val: float, vasopressor: bool) -> int:
    """MAP + vasopressor → SOFA cardiovascular score."""
    if vasopressor:
        return 3  # simplified: vasopressor use = score 3+
    if np.isnan(map_val):
        return np.nan
    if map_val >= 70:
        return 0
    else:
        return 1


def _sofa_neurological(gcs: float) -> int:
    """GCS → SOFA neurological score."""
    if np.isnan(gcs):
        return np.nan
    if gcs >= 15:
        return 0
    elif gcs >= 13:
        return 1
    elif gcs >= 10:
        return 2
    elif gcs >= 6:
        return 3
    else:
        return 4


def _sofa_renal(creatinine: float, uo_24h: float) -> int:
    """Creatinine (mg/dL) + urine output (mL/day) → SOFA renal score."""
    if not np.isnan(creatinine):
        if creatinine < 1.2:
            return 0
        elif creatinine < 2.0:
            return 1
        elif creatinine < 3.5:
            return 2
        elif creatinine < 5.0:
            return 3
        else:
            return 4
    if not np.isnan(uo_24h):
        if uo_24h < 200:
            return 4
        elif uo_24h < 500:
            return 3
    return np.nan


# ── Infection window detection ────────────────────────────────────────────────

def _get_infection_windows(raw_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Detect suspected infection time per hadm_id.
    Returns DataFrame: hadm_id, infection_time
    """
    hadm_ids = set(cohort["hadm_id"].tolist())

    # Blood cultures
    logger.info("Loading microbiologyevents...")
    micro = pd.read_csv(
        raw_dir / "hosp" / "microbiologyevents.csv.gz",
        compression="gzip",
        parse_dates=["charttime"],
        usecols=["hadm_id", "charttime", "spec_type_desc"],
    )
    micro = micro[
        micro["hadm_id"].isin(hadm_ids) &
        micro["spec_type_desc"].str.contains("BLOOD", case=False, na=False) &
        micro["charttime"].notna()
    ]
    micro = micro.rename(columns={"charttime": "culture_time"})

    # Antibiotics
    logger.info("Loading prescriptions (antibiotics)...")
    rx = pd.read_csv(
        raw_dir / "hosp" / "prescriptions.csv.gz",
        compression="gzip",
        parse_dates=["starttime"],
        usecols=["hadm_id", "drug", "starttime"],
    )
    rx = rx[
        rx["hadm_id"].isin(hadm_ids) &
        rx["drug"].apply(is_antibiotic) &
        rx["starttime"].notna()
    ]
    rx = rx.rename(columns={"starttime": "abx_time"})

    # Match: antibiotic-first (culture within 24h) or culture-first (abx within 72h)
    infection_times = []
    for hadm_id in tqdm(hadm_ids, desc="Infection window", leave=False):
        cultures = micro[micro["hadm_id"] == hadm_id]["culture_time"].tolist()
        abxs = rx[rx["hadm_id"] == hadm_id]["abx_time"].tolist()
        if not cultures or not abxs:
            continue

        t_inf = None
        for abx_t in abxs:
            for cult_t in cultures:
                delta_h = (cult_t - abx_t).total_seconds() / 3600.0
                # abx first, culture within 24h
                if 0 <= delta_h <= 24:
                    t_inf = abx_t
                    break
                # culture first, abx within 72h
                delta_h2 = (abx_t - cult_t).total_seconds() / 3600.0
                if 0 <= delta_h2 <= 72:
                    t_inf = cult_t
                    break
            if t_inf is not None:
                break

        if t_inf is not None:
            infection_times.append({"hadm_id": hadm_id, "infection_time": t_inf})

    return pd.DataFrame(infection_times)


# ── Hourly SOFA computation ───────────────────────────────────────────────────

def _compute_hourly_sofa(stay_id: int, stay_features: pd.DataFrame,
                         intime: pd.Timestamp, outtime: pd.Timestamp,
                         obs_window_h: int) -> pd.DataFrame:
    """
    Compute SOFA sub-scores and total at each hour of the ICU stay.
    Uses last-known value (forward-fill) within the stay.
    """
    hours = pd.date_range(
        start=intime.floor("1H"),
        end=min(outtime, intime + pd.Timedelta(hours=obs_window_h)),
        freq="1H",
    )

    # Pivot features to wide format, then forward-fill
    wide = (
        stay_features[stay_features["stay_id"] == stay_id]
        .pivot_table(index="charttime", columns="feature", values="value",
                     aggfunc="mean")
        .reindex(hours, method="ffill")
    )

    def get(col, default=np.nan):
        return wide[col].values if col in wide.columns else np.full(len(hours), default)

    pao2 = get("pao2")
    fio2_raw = get("fio2", 0.21)  # default room air
    fio2 = np.where(np.isnan(fio2_raw), 0.21, fio2_raw / 100.0)
    plts = get("platelets")
    bili = get("bilirubin")
    map_v = get("map")
    vaso = get("vasopressor", 0.0)
    gcs = get("gcs")
    creat = get("creatinine")
    uo = get("uo")

    rows = []
    for i, t in enumerate(hours):
        r = _sofa_respiratory(pao2[i], fio2[i])
        c = _sofa_coagulation(plts[i])
        l = _sofa_liver(bili[i])
        cv = _sofa_cardiovascular(map_v[i], bool(vaso[i] >= 0.5))
        n = _sofa_neurological(gcs[i])
        ren = _sofa_renal(creat[i], uo[i] * 24 if not np.isnan(uo[i]) else np.nan)

        scores = [r, c, l, cv, n, ren]
        valid = [s for s in scores if not (isinstance(s, float) and np.isnan(s))]
        total = sum(valid) if valid else np.nan

        rows.append({
            "stay_id": stay_id, "charttime": t,
            "sofa_resp": r, "sofa_coag": c, "sofa_liver": l,
            "sofa_cardio": cv, "sofa_neuro": n, "sofa_renal": ren,
            "sofa_total": total,
        })

    return pd.DataFrame(rows)


# ── Main label generation ─────────────────────────────────────────────────────

def generate_labels(cfg: dict, raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    """
    Generate Sepsis-3 labels for all ICU stays.
    """
    obs_window = cfg["data"]["observation_window"]

    cohort = pd.read_parquet(interim_dir / "cohort.parquet")
    features = pd.read_parquet(interim_dir / "features.parquet")
    logger.info(f"Cohort: {len(cohort):,}  Features: {len(features):,}")

    # Step 1: Infection windows
    infection_df = _get_infection_windows(raw_dir, cohort)
    hadm2inf = infection_df.set_index("hadm_id")["infection_time"].to_dict()
    logger.info(f"Stays with suspected infection: {len(hadm2inf):,}")

    cohort["infection_time"] = cohort["hadm_id"].map(hadm2inf)

    # Step 2 + 3: SOFA per stay + Sepsis onset
    all_sofa = []
    sepsis_info = []

    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Computing SOFA"):
        stay_id = row["stay_id"]
        intime = row["intime"]
        outtime = row["outtime"]
        inf_time = row["infection_time"]

        sofa_df = _compute_hourly_sofa(
            stay_id, features, intime, outtime, obs_window
        )
        all_sofa.append(sofa_df)

        # Baseline SOFA: minimum in first 24h
        first_24h = sofa_df[sofa_df["charttime"] <= intime + pd.Timedelta(hours=24)]
        baseline_sofa = first_24h["sofa_total"].min() if len(first_24h) > 0 else 0.0
        if np.isnan(baseline_sofa):
            baseline_sofa = 0.0

        # Sepsis onset: first time SOFA >= baseline + 2 AND infection window active
        sepsis_time = None
        if pd.notna(inf_time):
            # Infection window is active from infection_time onwards
            for _, srow in sofa_df.iterrows():
                t = srow["charttime"]
                if t < inf_time:
                    continue
                sofa = srow["sofa_total"]
                if not np.isnan(sofa) and sofa >= baseline_sofa + 2:
                    sepsis_time = t
                    break

        sepsis_info.append({
            "stay_id": stay_id,
            "sepsis_onset_time": sepsis_time,
            "has_sepsis": sepsis_time is not None,
            "baseline_sofa": baseline_sofa,
        })

    sofa_full = pd.concat(all_sofa, ignore_index=True)
    sepsis_df = pd.DataFrame(sepsis_info)

    logger.info(f"Sepsis prevalence: {sepsis_df['has_sepsis'].mean():.1%}")

    # Step 4: Merge SOFA with sepsis onset, create labels
    sofa_full = sofa_full.merge(sepsis_df, on="stay_id", how="left")

    for horizon in [6, 12, 24]:
        col = f"label_{horizon}h"
        sofa_full[col] = False
        mask = sofa_full["has_sepsis"]
        onset = sofa_full.loc[mask, "sepsis_onset_time"]
        t = sofa_full.loc[mask, "charttime"]
        delta_h = (onset - t).dt.total_seconds() / 3600.0
        sofa_full.loc[mask, col] = (delta_h > 0) & (delta_h <= horizon)

    # Step 5: Exclude patients with sepsis at admission (t=0)
    # If sepsis onset <= intime + 1h, exclude
    cohort_map = cohort.set_index("stay_id")["intime"].to_dict()
    sofa_full["intime"] = sofa_full["stay_id"].map(cohort_map)
    early_sepsis = set(
        sepsis_df[
            sepsis_df["has_sepsis"] &
            (sepsis_df["sepsis_onset_time"] <= sepsis_df["stay_id"].map(cohort_map) + pd.Timedelta(hours=1))
        ]["stay_id"].tolist()
    )
    logger.info(f"Excluding {len(early_sepsis)} stays with sepsis at admission")
    sofa_full = sofa_full[~sofa_full["stay_id"].isin(early_sepsis)]

    out_cols = ["stay_id", "charttime", "sofa_total", "sofa_resp", "sofa_coag",
                "sofa_liver", "sofa_cardio", "sofa_neuro", "sofa_renal",
                "label_6h", "label_12h", "label_24h", "has_sepsis", "sepsis_onset_time"]
    sofa_full = sofa_full[out_cols].sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    out_path = interim_dir / "labels.parquet"
    sofa_full.to_parquet(out_path, index=False)
    logger.info(f"Saved labels → {out_path}")

    for h in [6, 12, 24]:
        prev = sofa_full[f"label_{h}h"].mean()
        logger.info(f"  label_{h}h prevalence: {prev:.3f}")

    return sofa_full


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config("configs/base.yaml")
    generate_labels(
        cfg,
        raw_dir=Path(cfg["data"]["raw_dir"]),
        interim_dir=Path(cfg["data"]["interim_dir"]),
    )


if __name__ == "__main__":
    main()
