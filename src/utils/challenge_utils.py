"""
PhysioNet/CinC Challenge 2019 utility constants.

Feature set: 34 clinical features (8 vitals + 26 labs).
Demographics (Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS) are used only
for cohort metadata and splits — not as time-series features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Column names in PSV files ─────────────────────────────────────────────────
# These are the 40 columns in each patient's PSV file
ALL_PSV_COLUMNS = [
    # 8 vitals
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # 26 labs
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Fibrinogen", "Platelets",
    # 6 demographics/admin (not time-series features)
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    # Label
    "SepsisLabel",
]

# ── Feature ordering (index in the 34-dim feature vector) ────────────────────
FEATURE_NAMES = [
    # Vitals (0-7)
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # Labs (8-33)
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Fibrinogen", "Platelets",
]

N_FEATURES = len(FEATURE_NAMES)  # 34
FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

VITAL_FEATURES = FEATURE_NAMES[:8]
LAB_FEATURES = FEATURE_NAMES[8:]

# ── Outlier clipping ranges (physiologically valid) ───────────────────────────
CLIP_RANGES: dict[str, tuple[float, float]] = {
    "HR":              (0.0,    350.0),
    "O2Sat":           (0.0,    100.0),
    "Temp":            (25.0,   45.0),
    "SBP":             (0.0,    300.0),
    "MAP":             (0.0,    200.0),
    "DBP":             (0.0,    200.0),
    "Resp":            (0.0,    100.0),
    "EtCO2":           (0.0,    100.0),
    "BaseExcess":      (-30.0,  30.0),
    "HCO3":            (5.0,    60.0),
    "FiO2":            (0.21,   1.0),
    "pH":              (6.5,    8.0),
    "PaCO2":           (10.0,   150.0),
    "SaO2":            (0.0,    100.0),
    "AST":             (1.0,    5000.0),
    "BUN":             (1.0,    300.0),
    "Alkalinephos":    (1.0,    5000.0),
    "Calcium":         (0.5,    20.0),
    "Chloride":        (70.0,   150.0),
    "Creatinine":      (0.1,    30.0),
    "Bilirubin_direct":(0.1,    100.0),
    "Glucose":         (10.0,   2000.0),
    "Lactate":         (0.1,    30.0),
    "Magnesium":       (0.1,    10.0),
    "Phosphate":       (0.1,    20.0),
    "Potassium":       (1.0,    10.0),
    "Bilirubin_total": (0.1,    100.0),
    "TroponinI":       (0.0,    500.0),
    "Hct":             (5.0,    75.0),
    "Hgb":             (1.0,    25.0),
    "PTT":             (10.0,   200.0),
    "WBC":             (0.1,    500.0),
    "Fibrinogen":      (50.0,   2000.0),
    "Platelets":       (1.0,    2000.0),
}

# ── Recency thresholds τ_f (seconds) for 3-state missingness ─────────────────
# Vitals typically measured every 1-4h in ICU
# Labs typically ordered every 6-24h
RECENCY_THRESHOLD_INIT: dict[str, float] = {
    "HR":              7200.0,   # 2h
    "O2Sat":           7200.0,   # 2h
    "Temp":            14400.0,  # 4h
    "SBP":             7200.0,   # 2h
    "MAP":             7200.0,   # 2h
    "DBP":             7200.0,   # 2h
    "Resp":            7200.0,   # 2h
    "EtCO2":           14400.0,  # 4h (often ventilated patients only)
    "BaseExcess":      14400.0,  # 4h (ABG)
    "HCO3":            43200.0,  # 12h
    "FiO2":            14400.0,  # 4h
    "pH":              14400.0,  # 4h (ABG)
    "PaCO2":           14400.0,  # 4h (ABG)
    "SaO2":            14400.0,  # 4h
    "AST":             86400.0,  # 24h
    "BUN":             43200.0,  # 12h
    "Alkalinephos":    86400.0,  # 24h
    "Calcium":         43200.0,  # 12h
    "Chloride":        43200.0,  # 12h
    "Creatinine":      43200.0,  # 12h
    "Bilirubin_direct":86400.0,  # 24h
    "Glucose":         14400.0,  # 4h
    "Lactate":         14400.0,  # 4h
    "Magnesium":       43200.0,  # 12h
    "Phosphate":       43200.0,  # 12h
    "Potassium":       43200.0,  # 12h
    "Bilirubin_total": 86400.0,  # 24h
    "TroponinI":       86400.0,  # 24h
    "Hct":             43200.0,  # 12h
    "Hgb":             43200.0,  # 12h
    "PTT":             86400.0,  # 24h
    "WBC":             43200.0,  # 12h
    "Fibrinogen":      86400.0,  # 24h
    "Platelets":       43200.0,  # 12h
}


def get_recency_thresholds_array() -> np.ndarray:
    """Return τ_f initial values as a numpy array in FEATURE_NAMES order."""
    return np.array(
        [RECENCY_THRESHOLD_INIT[f] for f in FEATURE_NAMES],
        dtype=np.float32,
    )


def clip_feature(arr: np.ndarray, feature: str) -> np.ndarray:
    """Clip feature values to physiologically valid range."""
    lo, hi = CLIP_RANGES.get(feature, (-np.inf, np.inf))
    return np.clip(arr, lo, hi)
