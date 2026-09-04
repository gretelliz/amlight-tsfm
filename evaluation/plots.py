"""Figure generation shared by the experiment scripts.

Style follows IEEE conference paper conventions. Figures are saved as both
PDF (for LaTeX) and PNG (for quick inspection).

The paper's main figures (learning curves, Figs. 1-3) are produced by
plot_learning_curves.py at the repo root; this module holds the smaller
diagnostic figures generated inline by the experiment scripts themselves
(exp3_full_da.py): zero-shot vs. fine-tuned comparison bars and training
loss curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODEL_COLORS = {
    "rule_based": "#888888",
    "timesfm": "#2196F3",
    "moirai": "#4CAF50",
    "chronos": "#FF9800",
    "patchtst": "#9C27B0",
    "ttm": "#E91E63",
}

MODEL_LABELS = {
    "rule_based": "Rule-based",
    "timesfm": "TimesFM",
    "moirai": "Moirai",
    "chronos": "Chronos",
    "patchtst": "PatchTST",
    "ttm": "TTM/Granite",
}

# T1: Skill_SN: T2: AUROC; T3: Skill; T4: MAE (lower is better)
TASK_LABELS = {
    "t1_trajectory": "T1: Trajectory (Skill_SN)",
    "t2_exceedance": "T2: Exceedance (AUROC)",
    "t3_timing":     "T3: Peak Timing (Skill)",
    "t4_asym":       "T4: Asymmetry (MAE)",
}
LOWER_IS_BETTER_TASKS = {"t4_asym"}


def _save(fig: plt.Figure, name: str, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}/{name}.{{pdf,png}}")


def plot_experiment3_finetune(
    results_zeroshot: dict[str, dict[str, float]],
    results_finetuned: dict[str, dict[str, float]],
    out_dir: str | Path = "results/figures",
) -> None:
    """Grouped bar: zero-shot vs. Full-DA fine-tuned, one panel per task (T1-T4).

    Both dicts map model_name -> {task_key: scalar_metric}, as produced by
    exp3_full_da.py's `_metric()` helper.
    """
    models = [m for m in ("timesfm", "moirai", "chronos", "patchtst", "ttm")
              if m in results_zeroshot or m in results_finetuned]
    tasks  = list(TASK_LABELS)

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 3.0))
    for ax, task in zip(axes, tasks):
        lower_better = task in LOWER_IS_BETTER_TASKS
        x  = np.arange(len(models))
        zs = [results_zeroshot.get(m, {}).get(task, 0.0) for m in models]
        ft = [results_finetuned.get(m, {}).get(task, 0.0) for m in models]

        ax.bar(x - 0.2, zs, 0.38, label="Zero-shot", alpha=0.70,
               color=[MODEL_COLORS[m] for m in models], hatch="//")
        ax.bar(x + 0.2, ft, 0.38, label="Full DA", alpha=0.85,
               color=[MODEL_COLORS[m] for m in models])
        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("MAE  ↓ better" if lower_better else "Skill / AUROC  ↑ better")
        if not lower_better:
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Full DA — Zero-shot vs. Fine-tuned", y=1.02)
    fig.tight_layout()
    _save(fig, "exp3_fulldA", out_dir)


def plot_training_history(
    histories: dict[str, dict[str, list[float]]],
    out_dir: str | Path = "results/figures",
) -> None:
    """Training loss curves for all fine-tuned models."""
    n = len(histories)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.0))
    if n == 1:
        axes = [axes]

    for ax, (model, hist) in zip(axes, histories.items()):
        ax.plot(hist.get("train_loss", []), label="Train", color=MODEL_COLORS.get(model, "blue"))
        ax.plot(hist.get("val_loss", []), label="Val", color=MODEL_COLORS.get(model, "blue"),
                linestyle="--", alpha=0.7)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Full DA — Stage 1 Training Loss Curves", y=1.02)
    fig.tight_layout()
    _save(fig, "training_history", out_dir)
