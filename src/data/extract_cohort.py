"""
Step 1: Extract ICU stay cohort from MIMIC-IV.

Inclusion criteria:
  - First ICU stay per hospital admission
  - Age >= 18 at ICU admission
  - ICU length of stay >= min_stay_hours

Output: data/interim/cohort.parquet
Columns: subject_id, hadm_id, stay_id, intime, outtime,
         age, gender, ethnicity, first_careunit, los_hours
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def extract_cohort(cfg: dict, raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Extract and filter the ICU cohort.

    Args:
        cfg:     loaded config (base.yaml)
        raw_dir: path to MIMIC-IV raw CSV files
        out_dir: path to save interim parquet

    Returns:
        DataFrame with one row per included ICU stay
    """
    min_stay_hours = cfg["data"]["min_stay_hours"]

    logger.info("Loading icustays...")
    icu = pd.read_csv(
        raw_dir / "icu" / "icustays.csv.gz",
        compression="gzip",
        parse_dates=["intime", "outtime"],
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"],
    )

    logger.info("Loading admissions...")
    adm = pd.read_csv(
        raw_dir / "hosp" / "admissions.csv.gz",
        compression="gzip",
        parse_dates=["admittime", "dischtime"],
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "gender", "race"],
    )

    logger.info("Loading patients (for age)...")
    pts = pd.read_csv(
        raw_dir / "hosp" / "patients.csv.gz",
        compression="gzip",
        usecols=["subject_id", "anchor_age", "anchor_year", "gender"],
    )

    # Merge
    df = icu.merge(adm[["hadm_id", "admittime", "gender", "race"]], on="hadm_id", how="left")
    df = df.merge(pts[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left")

    # Compute age at ICU admission
    df["icu_year"] = df["intime"].dt.year
    df["age"] = df["anchor_age"] + (df["icu_year"] - df["anchor_year"])
    df.rename(columns={"race": "ethnicity"}, inplace=True)

    # Filter: age >= 18
    before = len(df)
    df = df[df["age"] >= 18].copy()
    logger.info(f"Age filter: {before} → {len(df)}")

    # Filter: first ICU stay per admission
    df = df.sort_values("intime").groupby("hadm_id").first().reset_index()
    logger.info(f"First ICU stay per admission: {len(df)}")

    # Compute LOS and filter: min_stay_hours
    df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    before = len(df)
    df = df[df["los_hours"] >= min_stay_hours].copy()
    logger.info(f"LOS >= {min_stay_hours}h filter: {before} → {len(df)}")

    # Keep relevant columns
    keep_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
                 "age", "gender", "ethnicity", "first_careunit", "los_hours"]
    df = df[keep_cols].reset_index(drop=True)

    logger.info(f"Final cohort: {len(df)} ICU stays")
    logger.info(f"ICU units:\n{df['first_careunit'].value_counts()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cohort.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved cohort → {out_path}")

    return df


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config("configs/base.yaml")
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["interim_dir"])
    extract_cohort(cfg, raw_dir, out_dir)


if __name__ == "__main__":
    main()
