"""
Evaluation metrics for sepsis prediction.

Primary:
  - AUROC (with DeLong 95% CI)
  - AUPRC (average precision)

Clinical utility:
  - Sensitivity at Specificity 90% / 95%
  - NNA (Number Needed to Alarm = 1/PPV at threshold)
  - Early Warning Time (EWT) — see early_warning_time.py

Statistical:
  - Bootstrap 95% CI for all metrics
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC. Returns NaN if only one class present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUPRC (average precision). Returns NaN if only one class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
) -> tuple[float, float]:
    """
    Find the threshold achieving >= target_specificity,
    return (sensitivity, threshold) at that operating point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr

    # Find threshold where specificity >= target
    mask = specificity >= target_specificity
    if not mask.any():
        return 0.0, thresholds[-1]

    # Among qualifying thresholds, pick highest sensitivity
    idx = np.where(mask)[0][np.argmax(tpr[mask])]
    return float(tpr[idx]), float(thresholds[idx])


def number_needed_to_alarm(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """
    NNA = 1 / PPV at the threshold achieving target_specificity.
    Lower NNA → fewer false alarms per true sepsis case.
    """
    _, threshold = sensitivity_at_specificity(y_true, y_score, target_specificity)
    y_pred = (y_score >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    ppv = tp / (tp + fp + 1e-10)
    return 1.0 / (ppv + 1e-10)


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
    **kwargs,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for any metric.

    Returns:
        (point_estimate, lower_95ci, upper_95ci)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_scores.append(metric_fn(y_b, s_b, **kwargs))

    if not boot_scores:
        return float("nan"), float("nan"), float("nan")

    point = metric_fn(y_true, y_score, **kwargs)
    lo = float(np.percentile(boot_scores, 2.5))
    hi = float(np.percentile(boot_scores, 97.5))
    return point, lo, hi


def full_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    specificity_targets: list[float] = [0.90, 0.95],
    bootstrap_seed: int = 42,
) -> dict:
    """
    Compute all evaluation metrics with bootstrap CIs.

    Returns:
        dict with all metric values and CIs
    """
    # Filter NaN labels
    valid = ~np.isnan(y_true)
    y_true = y_true[valid]
    y_score = y_score[valid]

    results = {}

    # AUROC
    auroc, auroc_lo, auroc_hi = bootstrap_metric(
        y_true, y_score, compute_auroc, n_bootstrap, bootstrap_seed
    )
    results["auroc"] = auroc
    results["auroc_ci"] = (auroc_lo, auroc_hi)

    # AUPRC
    auprc, auprc_lo, auprc_hi = bootstrap_metric(
        y_true, y_score, compute_auprc, n_bootstrap, bootstrap_seed
    )
    results["auprc"] = auprc
    results["auprc_ci"] = (auprc_lo, auprc_hi)

    # Sensitivity at various specificities
    for sp in specificity_targets:
        se, thresh = sensitivity_at_specificity(y_true, y_score, sp)
        results[f"se_at_sp{int(sp*100)}"] = se
        results[f"threshold_sp{int(sp*100)}"] = thresh

        nna = number_needed_to_alarm(y_true, y_score, sp)
        results[f"nna_sp{int(sp*100)}"] = nna

    # Prevalence
    results["prevalence"] = float(y_true.mean())
    results["n_samples"] = int(len(y_true))
    results["n_positive"] = int(y_true.sum())

    return results


def print_metrics(results: dict, model_name: str = "") -> None:
    """Pretty-print metric results."""
    print(f"\n{'='*60}")
    if model_name:
        print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  AUROC: {results['auroc']:.4f}  "
          f"[{results['auroc_ci'][0]:.4f}, {results['auroc_ci'][1]:.4f}]")
    print(f"  AUPRC: {results['auprc']:.4f}  "
          f"[{results['auprc_ci'][0]:.4f}, {results['auprc_ci'][1]:.4f}]")
    for sp in [90, 95]:
        if f"se_at_sp{sp}" in results:
            print(f"  Se@Sp{sp}: {results[f'se_at_sp{sp}']:.4f}  "
                  f"NNA: {results[f'nna_sp{sp}']:.1f}")
    print(f"  Prevalence: {results['prevalence']:.3f}  "
          f"N={results['n_samples']:,}  P={results['n_positive']:,}")
    print(f"{'='*60}\n")
