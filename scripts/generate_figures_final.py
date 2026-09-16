"""
Generate all paper figures from verified Kaggle experiment results.
No JSON files needed — all values hard-coded from confirmed output.

Usage:
    python scripts/generate_figures_final.py
Output: results/figures/  (PDF + PNG)
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
PURPLE = "#7b3294"
ORANGE = "#b35806"
GRAY   = "#969696"

MINUS  = "−"   # typographic minus used in figure labels

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")


def load_results(name: str) -> dict:
    """Read an analysis artifact.

    Figures must not carry their own numbers. A hard-coded value cannot
    disagree with the analysis loudly -- it just quietly becomes the number
    that gets published.
    """
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the matching scripts/analyze_*.py first; "
            f"figures are rendered from analysis output, not from literals."
        )
    with open(path) as f:
        return json.load(f)


def pick(data: dict, model: str, key: str, field: str = "auroc") -> float:
    """Fetch one metric, failing loudly rather than substituting a placeholder."""
    try:
        v = data[model][key][field]
    except (KeyError, TypeError):
        raise KeyError(
            f"{field} missing for model={model!r} key={key!r}; re-run the analysis"
        ) from None
    if v is None:
        raise KeyError(f"{field} is null for model={model!r} key={key!r}")
    return float(v)


# ── Figure 1: Subgroup AUROC ───────────────────────────────────────────────

def fig1_subgroup():
    subgroups = ["Overall", "Low\nmissingness", "High\nmissingness",
                 "Short\nstays", "Long\nstays"]
    keys = ["overall", "miss_low", "miss_high", "short_stay", "long_stay"]

    raw = load_results("subgroup_analysis.json")
    data = {
        model: [pick(raw, model, k) for k in keys]
        for model in ["IMST-Mamba", "Transformer", "GRU-D", "LSTM"]
    }
    colors = [BLUE, RED, GREEN, PURPLE]
    x = np.arange(len(subgroups))
    w = 0.18

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (model, vals) in enumerate(data.items()):
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=model,
                      color=colors[i], edgecolor="white", linewidth=0.5)

    # Highlight long-stays reversal
    long_idx = 4
    ax.annotate("IMST-Mamba\nsurpasses\nTransformer",
                xy=(long_idx - 1.5*w, 0.855),
                xytext=(long_idx - 0.65, 0.925),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
                fontsize=8, color=BLUE, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(subgroups)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.56, 0.99)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(ncol=2, framealpha=0.9, loc="upper left")
    ax.set_title("AUROC by Patient Subgroup")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig1_subgroup_auroc.pdf")
    fig.savefig(FIG_DIR / "fig1_subgroup_auroc.png")
    plt.close(fig)
    print("✓ fig1_subgroup_auroc")


# ── Figure 2: Ablation ─────────────────────────────────────────────────────

def fig2_ablation():
    labels = [
        "Full model",
        "− mask  ($m$=1)",
        "− staleness  ($s$=0)",
        "− inter-event  ($\\delta t$=1)",
    ]
    abl = load_results("ablation_results.json")
    aurocs = [pick({"m": abl}, "m", k)
              for k in ["full", "no_mask", "no_s", "no_delta_t"]]
    colors = [BLUE, "#f4a582", RED, "#d1e5f0"]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, aurocs, color=colors, edgecolor="white", linewidth=0.5)

    base = aurocs[0]
    deltas = [a - base for a in aurocs]
    tags = ["baseline"] + [
        f"{MINUS if d < 0 else chr(43)}{abs(d):.3f}" for d in deltas[1:]
    ]
    for bar, tag, d in zip(bars, tags, deltas):
        col = "white" if abs(d) > 0.01 or tag == "baseline" else "#333"
        ax.text(bar.get_width() - 0.002, bar.get_y() + bar.get_height()/2,
                tag, va="center", ha="right", color=col, fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.61, 0.81)
    ax.axvline(base, color=BLUE, linewidth=1.2, linestyle="--",
               label=f"Full model ({base:.4f})", alpha=0.8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ablation Study — IMST-Mamba")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig2_ablation.pdf")
    fig.savefig(FIG_DIR / "fig2_ablation.png")
    plt.close(fig)
    print("✓ fig2_ablation")


# ── Figure 3: Reliability diagram (MC Dropout) ────────────────────────────

def fig3_reliability():
    mc = load_results("mc_dropout_results.json")

    bins = [b for b in mc["reliability_diagram"] if b.get("acc") is not None]
    if not bins:
        raise ValueError(
            "reliability_diagram has no populated bins; re-run "
            "scripts/analyze_mc_dropout.py"
        )
    conf_bins = [b["conf"] for b in bins]
    acc_bins  = [b["acc"]  for b in bins]
    n_bins    = [b["n"]    for b in bins]
    ece       = float(mc["ece"])

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", zorder=1)

    sizes = [max(20, n/3000) for n in n_bins]
    sc = ax.scatter(conf_bins, acc_bins, s=sizes, c=n_bins,
                    cmap="Blues", edgecolors=BLUE, linewidths=0.8, zorder=3)
    ax.plot(conf_bins, acc_bins, color=BLUE, linewidth=1.5,
            label=f"IMST-Mamba (ECE={ece:.3f})", zorder=2)

    plt.colorbar(sc, ax=ax, label="# timesteps", shrink=0.85)
    ax.fill_between(conf_bins, conf_bins, acc_bins, alpha=0.08, color=RED)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Reliability Diagram")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig3_reliability.pdf")
    fig.savefig(FIG_DIR / "fig3_reliability.png")
    plt.close(fig)
    print("✓ fig3_reliability")


# ── Figure 4: Uncertainty vs Missingness ──────────────────────────────────

def fig4_uncertainty_miss():
    mc = load_results("mc_dropout_results.json")

    deciles = mc["uncertainty_vs_missingness"]
    if not deciles:
        raise ValueError(
            "uncertainty_vs_missingness is empty; re-run "
            "scripts/analyze_mc_dropout.py"
        )
    miss_vals = [d["mean_miss"] for d in deciles]
    unc_vals  = [d["mean_unc"]  for d in deciles]

    corr = mc.get("corr_miss_uncertainty")
    if corr is None:
        raise KeyError(
            "corr_miss_uncertainty missing from mc_dropout_results.json; "
            "re-run scripts/analyze_mc_dropout.py (it now persists this value)"
        )

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(miss_vals, unc_vals, "o-", color=BLUE, linewidth=2, markersize=6)
    ax.fill_between(miss_vals, unc_vals, alpha=0.12, color=BLUE)
    ax.set_xlabel("Mean missingness rate (per patient)")
    ax.set_ylabel("Epistemic uncertainty (std dev)")
    ax.set_title("Uncertainty vs. Missingness Rate")
    ax.text(0.05, 0.88, f"r = {MINUS if corr < 0 else chr(43)}{abs(corr):.3f}",
            transform=ax.transAxes,
            fontsize=9, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=BLUE, alpha=0.8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig4_uncertainty_miss.pdf")
    fig.savefig(FIG_DIR / "fig4_uncertainty_miss.png")
    plt.close(fig)
    print("✓ fig4_uncertainty_miss")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating figures → {FIG_DIR.resolve()}\n")
    fig1_subgroup()
    fig2_ablation()
    fig3_reliability()
    fig4_uncertainty_miss()
    print(f"\nDone. All figures in {FIG_DIR}/")
