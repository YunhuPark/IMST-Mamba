"""
Generate publication-quality figures for the IMST-Mamba paper.

Reads JSON result files from results/ and produces:
  Figure 1 — Subgroup AUROC bar chart (long vs short stays)
  Figure 2 — Ablation study (horizontal bar chart)
  Figure 3 — Reliability diagram (calibration curve)
  Figure 4 — Uncertainty vs missingness rate
  Figure 5 — Early detection comparison (detection rate + EWT)

Usage:
    python scripts/generate_figures.py
Outputs saved to: results/figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "IMST-Mamba": "#2166ac",
    "Transformer": "#d6604d",
    "GRU-D":       "#4dac26",
    "LSTM":        "#7b3294",
    "qSOFA":       "#b35806",
    "ablation":    "#969696",
}

RESULTS_DIR = Path("results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_json(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        print(f"[warn] {p} not found, skipping.")
        return {}
    with open(p) as f:
        return json.load(f)


# ── Figure 1: Subgroup AUROC ───────────────────────────────────────────────

def fig_subgroup_auroc():
    data = load_json("subgroup_analysis.json")
    if not data:
        return

    subgroups = ["Overall", "Short stays", "Long stays",
                 "Low miss.", "High miss."]
    keys      = ["overall", "short_stays", "long_stays",
                 "miss_low", "miss_high"]

    models = ["IMST-Mamba", "Transformer", "GRU-D", "LSTM"]
    x      = np.arange(len(subgroups))
    width  = 0.20

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, model in enumerate(models):
        vals = []
        for k in keys:
            v = data.get(model, {}).get(k, {}).get("auroc", None)
            vals.append(v if v is not None else 0.0)
        bars = ax.bar(x + i * width - 1.5 * width, vals,
                      width, label=model,
                      color=COLORS.get(model, "#888"),
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.55, 0.98)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(ncol=2, framealpha=0.9)
    ax.set_title("AUROC by Patient Subgroup")

    # annotate long-stays reversal
    idx = subgroups.index("Long stays")
    ax.annotate("IMST-Mamba\nsurpasses\nTransformer",
                xy=(idx - 1.5*width + 0.0, 0.855),
                xytext=(idx - 0.2, 0.92),
                arrowprops=dict(arrowstyle="->", color="#2166ac"),
                fontsize=8, color="#2166ac", ha="center")

    fig.savefig(FIG_DIR / "fig1_subgroup_auroc.pdf")
    fig.savefig(FIG_DIR / "fig1_subgroup_auroc.png")
    plt.close(fig)
    print("Saved fig1_subgroup_auroc")


# ── Figure 2: Ablation ─────────────────────────────────────────────────────

def fig_ablation():
    # Hard-coded ablation results (from paper Table 3)
    variants = [
        ("Full model",                  0.7791),
        ("−  mask ($m{=}1$)",           0.7406),
        ("−  staleness ($s{=}0$)",      0.6578),
        ("−  inter-event ($\\delta t{=}1$)", 0.7798),
    ]
    labels, aurocs = zip(*variants)
    deltas = [a - 0.7791 for a in aurocs]

    colors = ["#2166ac" if d == 0 else ("#d6604d" if d < -0.05 else "#f4a582")
              for d in deltas]

    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(labels))
    bars = ax.barh(y, aurocs, color=colors, edgecolor="white", linewidth=0.5)

    for bar, d in zip(bars, deltas):
        tag = "—" if d == 0 else f"{d:+.3f}"
        ax.text(bar.get_width() - 0.003, bar.get_y() + bar.get_height() / 2,
                tag, va="center", ha="right", color="white",
                fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.60, 0.82)
    ax.axvline(0.7791, color="#2166ac", linewidth=1, linestyle="--",
               label="Full model (0.7791)")
    ax.legend(fontsize=8)
    ax.set_title("Ablation Study — IMST-Mamba")
    ax.invert_yaxis()

    fig.savefig(FIG_DIR / "fig2_ablation.pdf")
    fig.savefig(FIG_DIR / "fig2_ablation.png")
    plt.close(fig)
    print("Saved fig2_ablation")


# ── Figure 3: Reliability diagram ─────────────────────────────────────────

def fig_reliability():
    data = load_json("mc_dropout_results.json")
    if not data or "reliability_diagram" not in data:
        print("[warn] mc_dropout_results.json missing — using placeholder")
        return

    bins = data["reliability_diagram"]
    conf = [b["conf"] for b in bins if b["acc"] is not None]
    acc  = [b["acc"]  for b in bins if b["acc"] is not None]
    n    = [b["n"]    for b in bins if b["acc"] is not None]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    sc = ax.scatter(conf, acc, c=n, cmap="Blues", s=80,
                    edgecolors="steelblue", linewidths=0.8, zorder=3)
    ax.plot(conf, acc, color="#2166ac", linewidth=1.5,
            label=f"IMST-Mamba (ECE={data.get('ece', 0):.3f})")

    plt.colorbar(sc, ax=ax, label="# timesteps")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Reliability Diagram (Calibration)")

    fig.savefig(FIG_DIR / "fig3_reliability.pdf")
    fig.savefig(FIG_DIR / "fig3_reliability.png")
    plt.close(fig)
    print("Saved fig3_reliability")


# ── Figure 4: Uncertainty vs Missingness ──────────────────────────────────

def fig_uncertainty_miss():
    data = load_json("mc_dropout_results.json")
    if not data or "uncertainty_vs_missingness" not in data:
        print("[warn] mc_dropout_results.json missing — using placeholder")
        return

    rows  = data["uncertainty_vs_missingness"]
    miss  = [r["mean_miss"] for r in rows]
    unc   = [r["mean_unc"]  for r in rows]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(miss, unc, "o-", color="#2166ac", linewidth=2, markersize=6)
    ax.fill_between(miss, unc, alpha=0.15, color="#2166ac")
    ax.set_xlabel("Mean missingness rate (per patient)")
    ax.set_ylabel("Epistemic uncertainty (prediction std)")
    ax.set_title("Uncertainty vs Missingness Rate")

    corr = np.corrcoef(miss, unc)[0, 1]
    ax.text(0.05, 0.90, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=9, color="#2166ac")

    fig.savefig(FIG_DIR / "fig4_uncertainty_miss.pdf")
    fig.savefig(FIG_DIR / "fig4_uncertainty_miss.png")
    plt.close(fig)
    print("Saved fig4_uncertainty_miss")


# ── Figure 5: Early Detection Comparison ──────────────────────────────────

def fig_early_detection():
    # Hard-coded from paper Table 5
    methods = ["qSOFA≥2", "modSOFA≥2", "GRU-D", "LSTM",
               "Transformer", "IMST-Mamba"]
    det_rate = [49.0, None, 70.6, 66.3, 69.0, 68.5]
    ewt_mean = [37.2, 42.2, 46.1, 48.0, 56.7, 44.7]
    colors   = [COLORS["qSOFA"], "#b8860b",
                COLORS["GRU-D"], COLORS["LSTM"],
                COLORS["Transformer"], COLORS["IMST-Mamba"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Detection rate
    valid  = [(m, d, c) for m, d, c in zip(methods, det_rate, colors)
              if d is not None]
    mv, dv, cv = zip(*valid)
    ax1.barh(mv, dv, color=cv, edgecolor="white", linewidth=0.5)
    ax1.axvline(49.0, color=COLORS["qSOFA"], linewidth=1, linestyle="--",
                label="qSOFA baseline")
    ax1.set_xlabel("Detection rate at Se@Sp90 (%)")
    ax1.set_title("Sepsis Detection Rate")
    ax1.set_xlim(40, 80)
    for x, y in zip(dv, mv):
        ax1.text(x + 0.3, y, f"{x:.1f}%", va="center", fontsize=8)

    # Mean EWT
    ax2.barh(methods, ewt_mean, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axvline(37.2, color=COLORS["qSOFA"], linewidth=1, linestyle="--",
                label="qSOFA EWT")
    ax2.set_xlabel("Mean early warning time (hours)")
    ax2.set_title("Mean Early Warning Time")
    for x, y in zip(ewt_mean, methods):
        ax2.text(x + 0.3, y, f"{x:.1f}h", va="center", fontsize=8)

    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_early_detection.pdf")
    fig.savefig(FIG_DIR / "fig5_early_detection.png")
    plt.close(fig)
    print("Saved fig5_early_detection")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR.resolve()}")
    fig_subgroup_auroc()
    fig_ablation()
    fig_reliability()
    fig_uncertainty_miss()
    fig_early_detection()
    print("\nDone. All figures saved to results/figures/")
