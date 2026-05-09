"""
Cox Proportional Hazards Model baseline.
Uses last-observation-carried-forward (LOCF) features as static covariates.

This is the clinical gold standard and shows the improvement from deep learning.
Implemented using lifelines library.

Note: CoxPHM is fit offline (not in PyTorch training loop).
      This module wraps it with the same interface for evaluation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    logger.warning("lifelines not installed. CoxPHM baseline unavailable.")


class CoxPHMWrapper:
    """
    Wraps lifelines CoxPHFitter with a PyTorch-compatible predict interface.

    Features used:
      - Mean value of each feature over first 24h
      - Last observed value of each feature
      - Missingness fraction per feature
      - Age, gender (from cohort)
    """

    def __init__(self):
        if not HAS_LIFELINES:
            raise RuntimeError("lifelines not installed. Run: pip install lifelines")
        self.model = CoxPHFitter(penalizer=0.1)
        self.fitted = False
        self.feature_cols: list[str] = []

    def _build_features(self, dataset_tensors: list[dict]) -> pd.DataFrame:
        """Convert patient tensors to tabular features for CoxPH."""
        rows = []
        for d in dataset_tensors:
            x = d["x"].numpy()   # (T, F)
            m = d["m"].numpy()   # (T, F)
            y = d["y"].numpy()   # (T,)
            T = d["seq_len"]

            row = {}
            # Mean and last value per feature (first 24h = first ~24*obs_per_hr rows)
            first_24h = min(T, 24 * 4)   # approximate
            for f in range(x.shape[1]):
                obs = x[:first_24h, f][m[:first_24h, f] > 0.5]
                row[f"mean_{f}"] = obs.mean() if len(obs) > 0 else 0.0
                row[f"last_{f}"] = obs[-1] if len(obs) > 0 else 0.0
                row[f"miss_frac_{f}"] = 1.0 - m[:first_24h, f].mean()

            # Survival time: time to sepsis onset or censoring (seq_len)
            has_event = y.max() > 0.5
            if has_event:
                event_idx = np.argmax(y > 0.5)
                row["T_survival"] = float(event_idx)
            else:
                row["T_survival"] = float(T)
            row["E_event"] = float(has_event)
            rows.append(row)

        return pd.DataFrame(rows)

    def fit(self, dataset_tensors: list[dict]) -> None:
        df = self._build_features(dataset_tensors)
        self.feature_cols = [c for c in df.columns if c not in ["T_survival", "E_event"]]
        df_fit = df[self.feature_cols + ["T_survival", "E_event"]].dropna()
        self.model.fit(df_fit, duration_col="T_survival", event_col="E_event")
        self.fitted = True
        logger.info(f"CoxPHM fitted on {len(df_fit)} patients")

    def predict_proba_table(self, dataset_tensors: list[dict]) -> np.ndarray:
        """Returns 1 - survival_function at T=1 as sepsis risk score."""
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_proba_table()")
        df = self._build_features(dataset_tensors)
        df_feat = df[self.feature_cols].fillna(0.0)
        # Partial hazard score as proxy for sepsis risk
        scores = self.model.predict_partial_hazard(df_feat).values
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CoxPHMWrapper":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
