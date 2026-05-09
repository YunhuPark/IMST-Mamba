"""
Early Warning Time (EWT) computation.

EWT = median time between first correct alarm and actual sepsis onset.
Higher EWT → model warns earlier → more time for clinical intervention.

Target: median EWT > 4 hours (headline clinical impact metric).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ewt(
    y_true_per_patient: list[np.ndarray],        # list of (T,) label arrays per patient
    y_score_per_patient: list[np.ndarray],       # list of (T,) probability arrays
    times_per_patient: list[np.ndarray],         # list of (T,) time-since-ICU-admission (hours)
    threshold: float,
    min_alarm_duration_steps: int = 1,           # how many consecutive steps above threshold
) -> dict:
    """
    Compute EWT across all sepsis-positive patients.

    Args:
        y_true_per_patient:  binary labels per patient
        y_score_per_patient: predicted probabilities per patient
        times_per_patient:   hours since ICU admission per observation event
        threshold:           alarm threshold (from specificity target)
        min_alarm_duration_steps: consecutive high-risk steps to count as alarm

    Returns:
        dict with 'median_ewt_hours', 'mean_ewt_hours', 'early_fraction',
                  'all_ewts', 'n_sepsis', 'n_alarmed'
    """
    ewts = []
    n_sepsis = 0
    n_alarmed = 0

    for y, score, times in zip(y_true_per_patient, y_score_per_patient, times_per_patient):
        T = len(y)
        has_sepsis = y.max() > 0.5
        if not has_sepsis:
            continue

        n_sepsis += 1

        # Sepsis onset time (first positive label)
        onset_idx = int(np.argmax(y > 0.5))
        onset_time = times[onset_idx]

        # First alarm: first time score crosses threshold (and stays for min_duration)
        alarm_time = None
        for t in range(T):
            if score[t] >= threshold:
                # Check consecutive
                end = min(t + min_alarm_duration_steps, T)
                if all(score[t:end] >= threshold):
                    alarm_time = times[t]
                    break

        if alarm_time is None:
            continue

        n_alarmed += 1
        ewt = onset_time - alarm_time   # positive = alarm before onset
        ewts.append(ewt)

    if not ewts:
        return {
            "median_ewt_hours": float("nan"),
            "mean_ewt_hours": float("nan"),
            "early_fraction": 0.0,
            "all_ewts": [],
            "n_sepsis": n_sepsis,
            "n_alarmed": 0,
        }

    ewts = np.array(ewts)
    early = (ewts > 0).mean()   # fraction with alarm before onset

    return {
        "median_ewt_hours": float(np.median(ewts)),
        "mean_ewt_hours": float(np.mean(ewts)),
        "p25_ewt_hours": float(np.percentile(ewts, 25)),
        "p75_ewt_hours": float(np.percentile(ewts, 75)),
        "early_fraction": float(early),
        "all_ewts": ewts.tolist(),
        "n_sepsis": n_sepsis,
        "n_alarmed": n_alarmed,
        "alarm_rate": n_alarmed / max(n_sepsis, 1),
    }


def compute_ewt_from_model_output(
    predictions: dict,    # {'stay_id': [...], 'scores': [...], 'labels': [...], 'times': [...]}
    threshold: float,
) -> dict:
    """
    Convenience wrapper for compute_ewt using model output format.

    predictions dict fields:
      stay_ids: list of stay_ids
      scores:   list of (T,) numpy arrays — predicted P(sepsis)
      labels:   list of (T,) numpy arrays — ground truth labels
      times:    list of (T,) numpy arrays — hours since ICU admission
    """
    return compute_ewt(
        y_true_per_patient=predictions["labels"],
        y_score_per_patient=predictions["scores"],
        times_per_patient=predictions["times"],
        threshold=threshold,
    )
