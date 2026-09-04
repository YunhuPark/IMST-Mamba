from __future__ import annotations

import numpy as np

DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3
MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0


def prediction_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    """PhysioNet/CinC 2019 per-patient utility for binary hourly predictions."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shape")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("predictions must be binary")

    if np.any(labels):
        is_septic = True
        # Challenge training labels already begin at clinical onset - 6 hours.
        t_sepsis = int(np.argmax(labels) - DT_OPTIMAL)
    else:
        is_septic = False
        t_sepsis = float("inf")

    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL

    total = 0.0
    for t, prediction in enumerate(predictions):
        if t > t_sepsis + DT_LATE:
            continue
        if is_septic and prediction:
            if t <= t_sepsis + DT_OPTIMAL:
                total += max(m1 * (t - t_sepsis) + b1, U_FP)
            else:
                total += m2 * (t - t_sepsis) + b2
        elif (not is_septic) and prediction:
            total += U_FP
        elif is_septic and (not prediction):
            if t > t_sepsis + DT_OPTIMAL:
                total += m3 * (t - t_sepsis) + b3
        else:
            total += U_TN
    return float(total)


def normalized_utility(
    patient_labels: list[np.ndarray],
    patient_probabilities: list[np.ndarray],
    threshold: float,
) -> float:
    if len(patient_labels) != len(patient_probabilities):
        raise ValueError("patient label/probability lists must have equal length")

    observed_total = 0.0
    best_total = 0.0
    inaction_total = 0.0
    for labels, probabilities in zip(patient_labels, patient_probabilities):
        labels = np.asarray(labels, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        predictions = (probabilities >= threshold).astype(int)
        observed_total += prediction_utility(labels, predictions)

        best = np.zeros(len(labels), dtype=int)
        if np.any(labels):
            t_sepsis = int(np.argmax(labels) - DT_OPTIMAL)
            start = max(0, t_sepsis + DT_EARLY)
            end = min(t_sepsis + DT_LATE + 1, len(labels))
            best[start:end] = 1
        best_total += prediction_utility(labels, best)
        inaction_total += prediction_utility(labels, np.zeros(len(labels), dtype=int))

    denominator = best_total - inaction_total
    if denominator == 0:
        return 0.0
    return float((observed_total - inaction_total) / denominator)


def choose_utility_threshold(
    patient_labels: list[np.ndarray],
    patient_probabilities: list[np.ndarray],
) -> float:
    if not patient_probabilities:
        return 0.5
    probabilities = np.concatenate([np.asarray(p, dtype=float) for p in patient_probabilities])
    candidates = np.unique(np.clip(probabilities, 0.001, 0.999))
    if len(candidates) > 80:
        candidates = np.quantile(candidates, np.linspace(0.02, 0.98, 60))

    best_threshold = 0.5
    best_utility = -np.inf
    for threshold in candidates:
        utility = normalized_utility(patient_labels, patient_probabilities, float(threshold))
        if utility > best_utility:
            best_utility = utility
            best_threshold = float(threshold)
    return best_threshold
