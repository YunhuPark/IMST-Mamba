"""
Calibration evaluation for clinical prediction models.

Expected Calibration Error (ECE) and reliability diagrams.
Well-calibrated model: predicted P(sepsis)=0.7 → ~70% truly develop sepsis.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, dict]:
    """
    Compute ECE and per-bin statistics.

    Returns:
        (ece, bins_dict)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n = mask.sum()
        if n == 0:
            bin_stats.append({"lo": lo, "hi": hi, "n": 0, "frac_pos": 0.0, "avg_prob": 0.0})
            continue

        frac_pos = y_true[mask].mean()
        avg_prob = y_prob[mask].mean()
        ece += (n / len(y_true)) * abs(frac_pos - avg_prob)
        bin_stats.append({"lo": lo, "hi": hi, "n": n, "frac_pos": frac_pos, "avg_prob": avg_prob})

    return ece, bin_stats


def plot_reliability_diagram(
    model_results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
    n_bins: int = 10,
) -> None:
    """
    Plot reliability diagrams for multiple models.

    Args:
        model_results: {model_name: (y_true, y_prob)}
        save_path:     optional path to save figure
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, (y_true, y_prob)) in zip(axes, model_results.items()):
        ece, bins = expected_calibration_error(y_true, y_prob, n_bins)
        frac_pos = [b["frac_pos"] for b in bins if b["n"] > 0]
        avg_prob = [b["avg_prob"] for b in bins if b["n"] > 0]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.bar(avg_prob, frac_pos, width=1.0/n_bins, alpha=0.6, color="steelblue",
               align="center", label=f"ECE={ece:.3f}")
        ax.plot(avg_prob, frac_pos, "o-", color="steelblue")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{name}\nECE = {ece:.4f}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
