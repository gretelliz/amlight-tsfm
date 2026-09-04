"""Learning curves: ZS baseline vs ZS+MLP@n per task (paper Figs. 1-3, §5.4).

T1 (trajectory) is excluded — the ZS backbone already produces the 48-step
trajectory natively, so there is no task head and no label-efficiency curve
for it (see exp4_fewshot.py). PatchTST is excluded from these curves: Stage 2
uses the pretrained ZS backbone as a fixed feature extractor, and PatchTST
has no pre-trained checkpoint.

Reads:
  results/exp2_results.json  → zero-shot baselines (flat reference lines)
  results/exp4_results.json  → ZS+MLP@n curves  (keys: *_mlp)

Usage:
  python plot_learning_curves.py
  python plot_learning_curves.py --out figures/learning_curves.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MODELS   = ["chronos", "timesfm", "moirai", "ttm"]
COLORS   = {"chronos": "#e41a1c", "timesfm": "#377eb8",
             "moirai": "#4daf4a", "ttm": "#ff7f00"}
LABELS   = {"chronos": "Chronos", "timesfm": "TimesFM",
             "moirai":  "Moirai",  "ttm": "TTM-R2"}


RESULTS  = Path("results")
EXP2     = RESULTS / "exp2_results.json"
EXP4     = RESULTS / "exp4_results.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _mlp_curve(model_curves: dict, key: str, metric: str) -> tuple[list, list, list]:
    """Return (ns, means, stds) from a task_curves list."""
    pts = model_curves.get(key, [])
    ns    = [p["n"]                          for p in pts]
    means = [p[f"{metric}_mean"]             for p in pts]
    stds  = [p.get(f"{metric}_std", 0.0)     for p in pts]
    return ns, means, stds


def _zs_scalar(exp2: dict, model: str, task: str, metric: str):
    """Extract single ZS scalar from exp2_results.json."""
    try:
        return exp2[model][task][metric]
    except (KeyError, TypeError):
        return None


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot(exp2: dict, exp4: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("ZS baseline vs ZS + MLP@n  (per task)", fontsize=13, fontweight="bold")

    ax_t2, ax_t3, ax_t4 = axes

    tasks = [
        (ax_t2, "t2_exceedance_mlp", "auroc", "T2 — Peak Exceedance  (AUROC ↑)",   "AUROC"),
        (ax_t3, "t3_timing_mlp",     "skill", "T3 — Peak Timing  (Skill ↑)",        "Timing Skill"),
        (ax_t4, "t4_asym_mlp",       "mae",   "T4 — Asymmetry  (MAE ↓)",            "MAE"),
    ]

    # ZS key mapping in exp2
    zs_keys = {
        "t2_exceedance_mlp": ("t2_exceedance", "t2_auroc"),
        "t3_timing_mlp":     ("t3_timing",     "t3_skill"),
        "t4_asym_mlp":       ("t4_asym",       "t4_mae"),
    }

    for ax, curve_key, metric, title, ylabel in tasks:
        invert = (metric == "mae")   # lower-is-better for T4

        plotted_any = False
        for model in MODELS:
            m_curves = exp4.get(model, {})
            ns, means, stds = _mlp_curve(m_curves, curve_key, metric)
            if not ns:
                continue

            c = COLORS[model]
            ns_arr    = np.array(ns)
            means_arr = np.array(means)
            stds_arr  = np.array(stds)

            ax.plot(ns_arr, means_arr, "o-", color=c, label=LABELS[model], linewidth=1.8, markersize=4)
            if stds_arr.any():
                ax.fill_between(ns_arr,
                                means_arr - stds_arr,
                                means_arr + stds_arr,
                                color=c, alpha=0.12)

            # ZS flat reference line (dashed, same color)
            zs_task, zs_metric = zs_keys[curve_key]
            zs_val = _zs_scalar(exp2, model, zs_task, zs_metric)
            if zs_val is not None:
                ax.axhline(zs_val, color=c, linestyle=":", linewidth=1.0, alpha=0.55)

            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("n labeled examples", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)

        if invert:
            # Lower is better — add downward arrow annotation
            ax.annotate("↓ better", xy=(0.98, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=7, color="gray")

    # Shared legend
    handles, labels_leg = ax_t2.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color="gray", linestyle=":", linewidth=1.2)]
    labels_leg += ["ZS (no head)"]
    fig.legend(handles, labels_leg, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/learning_curves.png")
    ap.add_argument("--results", default="results",
                    help="Directory containing exp2_results.json and exp4_results.json")
    args = ap.parse_args()

    results_dir = Path(args.results)
    exp2 = _load(results_dir / "exp2_results.json")
    exp4 = _load(results_dir / "exp4_results.json")

    if not exp4:
        print("WARNING: exp4_results.json not found or empty — no ZS+MLP@n curves to plot")

    plot(exp2, exp4, Path(args.out))
