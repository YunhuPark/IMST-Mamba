"""
MIMIC-IV utility constants and helper functions.
- itemid mappings for vitals and labs
- unit conversions (F→C, mg/dL checks, etc.)
- outlier clipping ranges
- antibiotic drug list for infection window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Feature ordering (index in the 25-dim feature vector) ────────────────────
FEATURE_NAMES = [
    # Vitals (0-8)
    "hr", "sbp", "dbp", "map", "rr", "spo2", "temp", "gcs", "uo",
    # Labs (9-23)
    "wbc", "creatinine", "lactate", "bilirubin", "platelets",
    "pao2", "ph", "sodium", "potassium", "glucose",
    "hemoglobin", "bicarbonate", "bun", "inr", "crp",
    # Other (24)
    "vasopressor",
]
N_FEATURES = len(FEATURE_NAMES)
FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

VITAL_FEATURES = FEATURE_NAMES[:9]
LAB_FEATURES = FEATURE_NAMES[9:24]

# ── MIMIC-IV chartevents itemids ─────────────────────────────────────────────
CHART_ITEMIDS: dict[str, list[int]] = {
    "hr":    [220045],
    "sbp":   [220179, 224167],
    "dbp":   [220180, 224643],
    "map":   [220052, 225312],
    "rr":    [220210, 224689],
    "spo2":  [220277],
    "temp":  [223762, 226329],   # Celsius
    "temp_f": [223761],          # Fahrenheit — needs conversion
    "gcs_eye":    [220739],
    "gcs_verbal": [223900],
    "gcs_motor":  [223901],
}

# Urine output — aggregated from outputevents
URINE_ITEMIDS = [
    40055, 43175, 40069, 40094, 40715, 40473,
    40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651,
]

# ── MIMIC-IV labevents itemids ────────────────────────────────────────────────
LAB_ITEMIDS: dict[str, list[int]] = {
    "wbc":         [51301],
    "creatinine":  [50912],
    "lactate":     [50813],
    "bilirubin":   [50885],
    "platelets":   [51265],
    "pao2":        [50821],
    "ph":          [50820],
    "sodium":      [50983],
    "potassium":   [50971],
    "glucose":     [50931],
    "hemoglobin":  [51222],
    "bicarbonate": [50882],
    "bun":         [51006],
    "inr":         [51237],
    "crp":         [50889],
}

# Vasopressors from inputevents
VASOPRESSOR_ITEMIDS = [221906, 221662, 221289, 222315]

# ── Outlier clipping ranges ───────────────────────────────────────────────────
CLIP_RANGES: dict[str, tuple[float, float]] = {
    "hr":          (10.0,   300.0),
    "sbp":         (40.0,   250.0),
    "dbp":         (20.0,   200.0),
    "map":         (20.0,   200.0),
    "rr":          (4.0,    60.0),
    "spo2":        (50.0,   100.0),
    "temp":        (25.0,   45.0),
    "gcs":         (3.0,    15.0),
    "uo":          (0.0,    2000.0),
    "wbc":         (0.1,    500.0),
    "creatinine":  (0.1,    30.0),
    "lactate":     (0.1,    30.0),
    "bilirubin":   (0.1,    80.0),
    "platelets":   (1.0,    2000.0),
    "pao2":        (20.0,   700.0),
    "ph":          (6.5,    8.0),
    "sodium":      (100.0,  180.0),
    "potassium":   (1.0,    10.0),
    "glucose":     (10.0,   2000.0),
    "hemoglobin":  (1.0,    25.0),
    "bicarbonate": (5.0,    60.0),
    "bun":         (1.0,    300.0),
    "inr":         (0.5,    15.0),
    "crp":         (0.0,    600.0),
    "vasopressor": (0.0,    1.0),
}

# ── Recency thresholds τ_f (seconds) for 3-state missingness initialization ──
# These are initial values; learned during training
RECENCY_THRESHOLD_INIT: dict[str, float] = {
    # Vitals: expect re-measurement within ~2 hours
    "hr": 7200.0, "sbp": 7200.0, "dbp": 7200.0, "map": 7200.0,
    "rr": 7200.0, "spo2": 7200.0, "temp": 14400.0, "gcs": 28800.0, "uo": 3600.0,
    # Labs: expect re-order within ~12 hours
    "wbc": 43200.0, "creatinine": 43200.0, "lactate": 14400.0,
    "bilirubin": 86400.0, "platelets": 43200.0, "pao2": 14400.0,
    "ph": 14400.0, "sodium": 43200.0, "potassium": 43200.0,
    "glucose": 14400.0, "hemoglobin": 43200.0, "bicarbonate": 43200.0,
    "bun": 43200.0, "inr": 86400.0, "crp": 86400.0,
    "vasopressor": 3600.0,
}

# ── Antibiotic list (IDSA systemic antibiotics only) ─────────────────────────
ANTIBIOTIC_KEYWORDS = [
    "vancomycin", "piperacillin", "meropenem", "imipenem", "cefepime",
    "ceftriaxone", "ceftazidime", "ciprofloxacin", "levofloxacin",
    "metronidazole", "ampicillin", "nafcillin", "oxacillin", "azithromycin",
    "trimethoprim", "daptomycin", "linezolid", "clindamycin", "gentamicin",
    "tobramycin", "amikacin", "aztreonam", "ertapenem", "colistin",
]

# ── Utility functions ─────────────────────────────────────────────────────────

def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def clip_feature(series: pd.Series, feature: str) -> pd.Series:
    """Clip feature values to physiologically valid range."""
    lo, hi = CLIP_RANGES.get(feature, (-np.inf, np.inf))
    return series.clip(lower=lo, upper=hi)


def is_antibiotic(drug_name: str) -> bool:
    """Check if a drug name (lowercase) matches the antibiotic list."""
    if not isinstance(drug_name, str):
        return False
    drug_lower = drug_name.lower()
    return any(kw in drug_lower for kw in ANTIBIOTIC_KEYWORDS)


def get_recency_thresholds_array() -> np.ndarray:
    """Return τ_f initial values as a numpy array in FEATURE_NAMES order."""
    return np.array(
        [RECENCY_THRESHOLD_INIT[f] for f in FEATURE_NAMES],
        dtype=np.float32,
    )
