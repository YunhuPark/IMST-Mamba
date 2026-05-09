"""
Statistical significance tests for AUROC comparison.

DeLong's method (Sun & Xu, 2014) for comparing two correlated AUROCs.
Bonferroni correction for multiple comparisons.

This file determines publication viability — significant improvement
over GRU-D with p < 0.00625 (Bonferroni corrected for 8 comparisons).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _structural_components(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute structural components for DeLong's method.
    Returns (V10, V01) — influence functions for positives and negatives.
    """
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]

    n1 = len(positives)
    n0 = len(negatives)

    # V10[i] = fraction of negatives that model correctly ranks below positive i
    # V01[j] = fraction of positives that model correctly ranks above negative j
    V10 = np.zeros(n1)
    V01 = np.zeros(n0)

    for i, p in enumerate(positives):
        V10[i] = np.mean(
            (p > negatives) + 0.5 * (p == negatives)
        )

    for j, n in enumerate(negatives):
        V01[j] = np.mean(
            (positives > n) + 0.5 * (positives == n)
        )

    return V10, V01


def delong_roc_test(
    y_true: np.ndarray,
    y_score_1: np.ndarray,
    y_score_2: np.ndarray,
) -> tuple[float, float, float]:
    """
    DeLong's test for comparing two correlated AUROCs.

    Tests H0: AUROC(model1) == AUROC(model2)
    Uses the exact covariance structure (Sun & Xu, 2014, Academic Radiology).

    Args:
        y_true:    ground truth binary labels
        y_score_1: scores from model 1 (proposed)
        y_score_2: scores from model 2 (baseline)

    Returns:
        (auroc_1, auroc_2, p_value)
    """
    V10_1, V01_1 = _structural_components(y_true, y_score_1)
    V10_2, V01_2 = _structural_components(y_true, y_score_2)

    auroc_1 = V10_1.mean()
    auroc_2 = V10_2.mean()

    n1 = len(V10_1)
    n0 = len(V01_1)

    # Covariance matrix of (AUROC_1, AUROC_2)
    S = np.zeros((2, 2))
    S[0, 0] = ((V10_1 - auroc_1) ** 2).sum() / (n1 - 1) / n1 + \
              ((V01_1 - auroc_1) ** 2).sum() / (n0 - 1) / n0

    S[1, 1] = ((V10_2 - auroc_2) ** 2).sum() / (n1 - 1) / n1 + \
              ((V01_2 - auroc_2) ** 2).sum() / (n0 - 1) / n0

    S[0, 1] = S[1, 0] = (
        np.cov(V10_1, V10_2)[0, 1] / n1 +
        np.cov(V01_1, V01_2)[0, 1] / n0
    )

    # Test statistic: (AUROC_1 - AUROC_2)^2 / var(AUROC_1 - AUROC_2)
    delta = auroc_1 - auroc_2
    var_delta = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_delta <= 0:
        return auroc_1, auroc_2, 1.0

    z = delta / np.sqrt(var_delta)
    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return auroc_1, auroc_2, float(p_value)


def compare_all_baselines(
    y_true: np.ndarray,
    proposed_scores: np.ndarray,
    baseline_scores: dict[str, np.ndarray],
    alpha: float = 0.05,
    n_comparisons: int | None = None,
) -> dict:
    """
    Run DeLong's test comparing proposed model against all baselines.
    Apply Bonferroni correction.

    Args:
        y_true:          ground truth
        proposed_scores: scores from proposed model
        baseline_scores: {model_name: scores} dict
        alpha:           significance level (before correction)
        n_comparisons:   number of comparisons for Bonferroni (default: len(baseline_scores))

    Returns:
        dict with comparison results
    """
    if n_comparisons is None:
        n_comparisons = len(baseline_scores)
    corrected_alpha = alpha / n_comparisons

    results = {}
    for name, scores in baseline_scores.items():
        auroc_proposed, auroc_baseline, p_val = delong_roc_test(
            y_true, proposed_scores, scores
        )
        results[name] = {
            "auroc_proposed": auroc_proposed,
            "auroc_baseline": auroc_baseline,
            "delta_auroc": auroc_proposed - auroc_baseline,
            "p_value": p_val,
            "p_corrected": min(p_val * n_comparisons, 1.0),
            "significant": p_val < corrected_alpha,
            "corrected_alpha": corrected_alpha,
        }

    return results


def print_comparison_table(results: dict) -> None:
    """Print formatted comparison table."""
    print(f"\n{'Model':<20} {'AUROC_Proposed':>15} {'AUROC_Base':>12} {'ΔAUROC':>8} {'p (corrected)':>15} {'Sig':>5}")
    print("-" * 80)
    for name, r in results.items():
        sig = "✓" if r["significant"] else ""
        print(
            f"{name:<20} {r['auroc_proposed']:>15.4f} "
            f"{r['auroc_baseline']:>12.4f} {r['delta_auroc']:>+8.4f} "
            f"{r['p_corrected']:>15.4f} {sig:>5}"
        )
    print()
