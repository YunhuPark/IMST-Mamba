"""
Step 5: Patient-level stratified train/val/test splits.

Split is done at subject_id level to prevent data leakage
(same patient cannot appear in both train and test).

Stratification key: sepsis_flag × age_group × icu_type

Output: data/processed/splits.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def make_splits(cfg: dict, interim_dir: Path, processed_dir: Path) -> dict:
    cohort = pd.read_parquet(interim_dir / "cohort.parquet")
    labels = pd.read_parquet(interim_dir / "labels.parquet")

    # Load valid stay ids (built by build_timeseries)
    with open(processed_dir / "valid_stay_ids.json") as f:
        valid_stay_ids = set(json.load(f))

    cohort = cohort[cohort["stay_id"].isin(valid_stay_ids)].copy()

    # Sepsis flag per stay
    sepsis_flag = (
        labels.groupby("stay_id")["has_sepsis"].max()
        .reindex(cohort["stay_id"])
        .fillna(False)
        .astype(int)
        .values
    )
    cohort["sepsis_flag"] = sepsis_flag

    # Age group
    cohort["age_group"] = pd.cut(
        cohort["age"], bins=[0, 50, 70, 200], labels=[0, 1, 2]
    ).astype(int)

    # ICU type (top 4 categories → 0-3, rest = 4)
    top_icus = cohort["first_careunit"].value_counts().head(4).index.tolist()
    cohort["icu_type"] = cohort["first_careunit"].apply(
        lambda x: top_icus.index(x) if x in top_icus else 4
    )

    # Composite stratification key
    cohort["strat_key"] = (
        cohort["sepsis_flag"].astype(str) + "_" +
        cohort["age_group"].astype(str) + "_" +
        cohort["icu_type"].astype(str)
    )

    # Split at subject_id level
    # First get one row per subject_id (use majority vote for strat_key)
    subj = (
        cohort.groupby("subject_id")
        .agg(strat_key=("strat_key", lambda x: x.mode()[0]))
        .reset_index()
    )

    train_frac = cfg["data"]["train_frac"]
    val_frac = cfg["data"]["val_frac"]
    test_frac = cfg["data"]["test_frac"]

    # First split: train vs (val+test)
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_frac + test_frac,
        random_state=cfg["seed"],
    )
    train_subj_idx, valtest_subj_idx = next(
        splitter1.split(subj, subj["strat_key"])
    )

    train_subj = subj.iloc[train_subj_idx]["subject_id"].tolist()
    valtest_subj = subj.iloc[valtest_subj_idx]

    # Second split: val vs test
    relative_test = test_frac / (val_frac + test_frac)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test,
        random_state=cfg["seed"],
    )
    val_subj_idx, test_subj_idx = next(
        splitter2.split(valtest_subj, valtest_subj["strat_key"])
    )
    val_subj = valtest_subj.iloc[val_subj_idx]["subject_id"].tolist()
    test_subj = valtest_subj.iloc[test_subj_idx]["subject_id"].tolist()

    # Map subject_id → stay_ids
    subj2stays = cohort.groupby("subject_id")["stay_id"].apply(list).to_dict()

    def get_stays(subject_ids):
        stays = []
        for s in subject_ids:
            stays.extend(subj2stays.get(s, []))
        return stays

    splits = {
        "train": get_stays(train_subj),
        "val": get_stays(val_subj),
        "test": get_stays(test_subj),
    }

    logger.info(f"Train stays: {len(splits['train']):,}")
    logger.info(f"Val   stays: {len(splits['val']):,}")
    logger.info(f"Test  stays: {len(splits['test']):,}")

    # Validate no subject leakage
    train_set = set(train_subj)
    val_set = set(val_subj)
    test_set = set(test_subj)
    assert not (train_set & val_set), "Subject leakage: train ∩ val"
    assert not (train_set & test_set), "Subject leakage: train ∩ test"
    assert not (val_set & test_set), "Subject leakage: val ∩ test"

    out_path = processed_dir / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits → {out_path}")

    return splits


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config("configs/base.yaml")
    make_splits(
        cfg,
        interim_dir=Path(cfg["data"]["interim_dir"]),
        processed_dir=Path(cfg["data"]["processed_dir"]),
    )


if __name__ == "__main__":
    main()
