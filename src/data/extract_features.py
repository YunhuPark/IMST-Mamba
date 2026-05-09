"""
Step 2: Extract vital signs and lab values from MIMIC-IV for each ICU stay.

Sources:
  - icu/chartevents  → vitals (HR, BP, RR, SpO2, Temp, GCS)
  - hosp/labevents   → labs (WBC, Creatinine, Lactate, ...)
  - icu/outputevents → urine output (hourly aggregation)
  - icu/inputevents  → vasopressor flag

Output: data/interim/features.parquet
Columns: stay_id, charttime, feature_name, value
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.mimic_utils import (
    CHART_ITEMIDS, LAB_ITEMIDS, URINE_ITEMIDS, VASOPRESSOR_ITEMIDS,
    FEATURE_IDX, fahrenheit_to_celsius, clip_feature,
)

logger = logging.getLogger(__name__)


def _extract_vitals(raw_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract chartevents for vital signs."""
    logger.info("Loading chartevents (this may take a few minutes)...")

    all_itemids = []
    for feat, ids in CHART_ITEMIDS.items():
        all_itemids.extend(ids)
    itemid_set = set(all_itemids)

    stay_ids = set(cohort["stay_id"].tolist())

    chunks = []
    for chunk in pd.read_csv(
        raw_dir / "icu" / "chartevents.csv.gz",
        compression="gzip",
        parse_dates=["charttime"],
        usecols=["stay_id", "itemid", "charttime", "valuenum", "valueuom"],
        chunksize=5_000_000,
    ):
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids) &
            chunk["itemid"].isin(itemid_set) &
            chunk["valuenum"].notna()
        ]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    logger.info(f"Chartevents rows after filter: {len(df):,}")

    # Build itemid → feature name map
    itemid2feat: dict[int, str] = {}
    for feat, ids in CHART_ITEMIDS.items():
        for iid in ids:
            itemid2feat[iid] = feat

    df["feature"] = df["itemid"].map(itemid2feat)

    # Fahrenheit → Celsius conversion
    mask_f = df["feature"] == "temp_f"
    df.loc[mask_f, "valuenum"] = df.loc[mask_f, "valuenum"].apply(fahrenheit_to_celsius)
    df.loc[mask_f, "feature"] = "temp"

    # GCS: sum eye + verbal + motor, grouped by stay_id + charttime
    gcs_mask = df["feature"].isin(["gcs_eye", "gcs_verbal", "gcs_motor"])
    gcs_df = df[gcs_mask].copy()
    gcs_sum = (
        gcs_df.groupby(["stay_id", "charttime"])["valuenum"]
        .sum()
        .reset_index()
        .rename(columns={"valuenum": "value"})
    )
    gcs_sum["feature"] = "gcs"
    df = df[~gcs_mask].rename(columns={"valuenum": "value"})
    df = pd.concat([df[["stay_id", "charttime", "feature", "value"]], gcs_sum], ignore_index=True)

    return df


def _extract_labs(raw_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract labevents."""
    logger.info("Loading labevents...")

    all_itemids = []
    for ids in LAB_ITEMIDS.values():
        all_itemids.extend(ids)
    itemid_set = set(all_itemids)

    hadm_ids = set(cohort["hadm_id"].tolist())
    # Join cohort to get stay_id and time window
    hadm2stay = cohort.set_index("hadm_id")[["stay_id", "intime", "outtime"]].to_dict("index")

    chunks = []
    for chunk in pd.read_csv(
        raw_dir / "hosp" / "labevents.csv.gz",
        compression="gzip",
        parse_dates=["charttime"],
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=5_000_000,
    ):
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids) &
            chunk["itemid"].isin(itemid_set) &
            chunk["valuenum"].notna()
        ]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    logger.info(f"Labevents rows after filter: {len(df):,}")

    # Map hadm_id → stay_id, filter to ICU window
    df["stay_id"] = df["hadm_id"].map(lambda h: hadm2stay.get(h, {}).get("stay_id"))
    df["intime"] = df["hadm_id"].map(lambda h: hadm2stay.get(h, {}).get("intime"))
    df["outtime"] = df["hadm_id"].map(lambda h: hadm2stay.get(h, {}).get("outtime"))
    df = df.dropna(subset=["stay_id"])
    df = df[(df["charttime"] >= df["intime"]) & (df["charttime"] <= df["outtime"] + pd.Timedelta(hours=24))]

    # itemid → feature name
    itemid2feat: dict[int, str] = {}
    for feat, ids in LAB_ITEMIDS.items():
        for iid in ids:
            itemid2feat[iid] = feat
    df["feature"] = df["itemid"].map(itemid2feat)

    return df[["stay_id", "charttime", "feature", "valuenum"]].rename(columns={"valuenum": "value"})


def _extract_urine(raw_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract and aggregate urine output per hour."""
    logger.info("Loading outputevents (urine)...")

    stay_ids = set(cohort["stay_id"].tolist())

    df = pd.read_csv(
        raw_dir / "icu" / "outputevents.csv.gz",
        compression="gzip",
        parse_dates=["charttime"],
        usecols=["stay_id", "itemid", "charttime", "value"],
    )
    df = df[df["stay_id"].isin(stay_ids) & df["itemid"].isin(URINE_ITEMIDS) & df["value"].notna()]

    # Aggregate to hourly urine output
    df["hour"] = df["charttime"].dt.floor("1H")
    uo = df.groupby(["stay_id", "hour"])["value"].sum().reset_index()
    uo.rename(columns={"hour": "charttime", "value": "value"}, inplace=True)
    uo["feature"] = "uo"
    return uo[["stay_id", "charttime", "feature", "value"]]


def _extract_vasopressors(raw_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract vasopressor flag (binary, hourly)."""
    logger.info("Loading inputevents (vasopressors)...")

    stay_ids = set(cohort["stay_id"].tolist())

    df = pd.read_csv(
        raw_dir / "icu" / "inputevents.csv.gz",
        compression="gzip",
        parse_dates=["starttime", "endtime"],
        usecols=["stay_id", "itemid", "starttime", "endtime"],
    )
    df = df[df["stay_id"].isin(stay_ids) & df["itemid"].isin(VASOPRESSOR_ITEMIDS)]

    # Expand to hourly rows
    rows = []
    for _, row in df.iterrows():
        start = row["starttime"].floor("1H")
        end = row["endtime"].ceil("1H") if pd.notna(row["endtime"]) else start + pd.Timedelta(hours=1)
        hours = pd.date_range(start, end, freq="1H")
        for h in hours:
            rows.append({"stay_id": row["stay_id"], "charttime": h, "feature": "vasopressor", "value": 1.0})

    if not rows:
        return pd.DataFrame(columns=["stay_id", "charttime", "feature", "value"])

    vp = pd.DataFrame(rows).drop_duplicates(subset=["stay_id", "charttime"])
    return vp


def extract_features(cfg: dict, raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    """
    Extract all features for all ICU stays.
    Returns long-format DataFrame: (stay_id, charttime, feature, value)
    """
    cohort = pd.read_parquet(interim_dir / "cohort.parquet")
    logger.info(f"Cohort size: {len(cohort):,} stays")

    vitals = _extract_vitals(raw_dir, cohort)
    labs = _extract_labs(raw_dir, cohort)
    uo = _extract_urine(raw_dir, cohort)
    vp = _extract_vasopressors(raw_dir, cohort)

    df = pd.concat([vitals, labs, uo, vp], ignore_index=True)
    logger.info(f"Total feature rows before clipping: {len(df):,}")

    # Clip outliers per feature
    clipped_parts = []
    for feat, grp in df.groupby("feature"):
        grp = grp.copy()
        grp["value"] = clip_feature(grp["value"], feat)
        clipped_parts.append(grp)
    df = pd.concat(clipped_parts, ignore_index=True)

    # Sort
    df = df.sort_values(["stay_id", "charttime", "feature"]).reset_index(drop=True)

    out_path = interim_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved features → {out_path}  ({len(df):,} rows)")

    return df


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config("configs/base.yaml")
    extract_features(
        cfg,
        raw_dir=Path(cfg["data"]["raw_dir"]),
        interim_dir=Path(cfg["data"]["interim_dir"]),
    )


if __name__ == "__main__":
    main()
