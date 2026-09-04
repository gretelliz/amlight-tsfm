"""Post-hoc statistical significance analysis (paper §4.1).

Compares:
  (A) Best classical baseline (exp1) vs Zero-Shot TSFMs (exp2)
  (B) Best classical baseline (exp1) vs LP-FT (exp3_lpft, frozen backbone + MLP)

Tasks
-----
  T1 (Trajectory Forecasting):      Skill_SN (higher = better)
  T2 (Peak Exceedance Detection):   AUROC    (higher = better)
  T3 (Peak Timing Prediction):      Skill + MAE(h) (higher skill / lower MAE = better)
  T4 (Traffic Asymmetry):           MAE (lower = better)

Bonferroni correction: 4 tasks → α_adj = 0.05/4 = 0.0125

Effect size: Cohen's d on per-window error differences (when available).

Run after all experiments:
    python compute_significance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results")
TASKS = ["t1_trajectory", "t2_exceedance", "t3_timing", "t4_asym"]
TASK_LABELS = {
    "t1_trajectory": "T1: Trajectory Forecasting (Skill_SN)",
    "t2_exceedance": "T2: Peak Exceedance Detection (AUROC)",
    "t3_timing":     "T3: Peak Timing Prediction (Skill / MAE)",
    "t4_asym":       "T4: Traffic Asymmetry Forecasting (MAE)",
}
LOWER_IS_BETTER = {"t4_asym"}
SKILL_TASKS = {"t1_trajectory", "t3_timing"}   # report Skill (T2 reports AUROC, T4 reports MAE)
# Tasks whose per-window error is squared error (vs absolute error) — affects
# which array compute_significance looks for when doing the paired Wilcoxon test.
_SE_WINDOW_TASKS = {"t2_exceedance"}
ALPHA = 0.05
N_TASKS = 4
ALPHA_BONF = ALPHA / N_TASKS   # 0.0125


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ── Helpers to extract the best result from the nested dicts ─────────────────

def _metric_val_ci(task: str, d: dict) -> tuple[float | None, list | None]:
    """Extract (primary_metric, CI) — Skill_SN for T1, AUROC for T2, Skill for T3, MAE for T4."""
    if task == "t1_trajectory":
        skill = d.get("t1_skill_sn")
        mase_ci = d.get("t1_mase_ci")
        ci = [1.0 - mase_ci[1], 1.0 - mase_ci[0]] if mase_ci else None
        return skill, ci
    elif task == "t2_exceedance":
        return d.get("t2_auroc"), d.get("t2_auroc_ci")
    elif task == "t3_timing":
        # primary = skill (higher = better); CI approximated from the MAE CI
        return d.get("t3_skill"), d.get("mae_ci")
    else:  # t4_asym
        return d.get("t4_mae"), d.get("t4_mae_ci")


def _best_baseline(exp1: dict, task: str) -> tuple[str, float, list | None]:
    """Best rule-based baseline: T1/T2/T3 → highest metric; T4 → lowest MAE."""
    task_data = exp1.get(task, {})
    higher = task not in LOWER_IS_BETTER
    best_val = -1.0 if higher else float("inf")
    best_name, best_ci = "", None
    for bkey, bdata in task_data.items():
        if not isinstance(bdata, dict):
            continue
        v, ci = _metric_val_ci(task, bdata)
        if v is None:
            continue
        if higher and v > best_val:
            best_val, best_name, best_ci = v, bkey, ci
        elif not higher and v < best_val:
            best_val, best_name, best_ci = v, bkey, ci
    return best_name, best_val, best_ci


def _best_model(exp: dict, task: str) -> tuple[str, float, list | None]:
    """Best TSFM in the given exp dict on this task."""
    higher = task not in LOWER_IS_BETTER
    best_val = -1.0 if higher else float("inf")
    best_name, best_ci = "", None
    for model_name, tasks in exp.items():
        v, ci = _metric_val_ci(task, tasks.get(task, {}))
        if v is None:
            continue
        if higher and v > best_val:
            best_val, best_name, best_ci = v, model_name, ci
        elif not higher and v < best_val:
            best_val, best_name, best_ci = v, model_name, ci
    return best_name, best_val, best_ci


# ── Per-window error arrays ───────────────────────────────────────────────────

def _load_window_errors(results: dict, task: str) -> dict[str, np.ndarray]:
    """Load per-window error arrays for paired significance tests.

    T2 (exceedance): uses "se_windows" (squared errors on the exceedance score).
    T1/T3 (trajectory, timing): use "ae_windows" (absolute errors).

    In both cases the Wilcoxon test checks errors_baseline > errors_model,
    so the direction is correct regardless of metric.
    """
    arrays: dict[str, np.ndarray] = {}
    exp1 = results.get("exp1", {})
    exp3 = results.get("exp3", {})   # Full DA — see run_all.py's merge key
    err_key = "se_windows" if task in _SE_WINDOW_TASKS else "ae_windows"

    for bkey, bdata in exp1.get(task, {}).items():
        if isinstance(bdata, dict) and err_key in bdata:
            arrays[f"baseline:{bkey}"] = np.array(bdata[err_key])

    for model_name, task_dict in exp3.items():
        d = task_dict.get(task, {})
        if err_key in d:
            arrays[f"ft:{model_name}"] = np.array(d[err_key])

    return arrays


def _wilcoxon_paired(errors_baseline: np.ndarray,
                     errors_model: np.ndarray) -> tuple[float, float, float]:
    """Wilcoxon signed-rank test on per-window error differences.

    H0: errors_baseline - errors_model has zero median (no improvement).
    H1 (one-sided): errors_baseline > errors_model (model is better).

    Returns (statistic, p_value, cohen_d).
    """
    diff = errors_baseline - errors_model   # positive = model is better
    if len(diff) < 10 or np.all(diff == 0):
        return float("nan"), 1.0, 0.0

    stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
    d = float(np.mean(diff) / (np.std(diff) + 1e-10))
    return float(stat), float(p), d


def _ci_overlap(ci_a: list | None, ci_b: list | None) -> str:
    if ci_a is None or ci_b is None:
        return "CI unavailable"
    overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
    return "overlap (not significant)" if overlap else "non-overlap (significant)"


# ── Main ─────────────────────────────────────────────────────────────────────

def _ci_str(ci, fmt=".3f"):
    if ci is None:
        return ""
    return f"  [{ci[0]:{fmt}}, {ci[1]:{fmt}}]"


def _print_task(task: str, b_name: str, b_val: float, b_ci,
                zs_name: str, zs_val: float, zs_ci,
                ft_name: str, ft_val: float, ft_ci,
                all_results: dict) -> tuple:
    higher = task not in LOWER_IS_BETTER
    print(f"\n── {TASK_LABELS[task]} ──")
    metric_label = "AUROC" if task == "t2_exceedance" else "Skill" if task in SKILL_TASKS else "MAE"
    sign_str     = "(↑ better)" if higher else "(↓ better)"

    def delta(a, b): return (a - b) if higher else (b - a)

    print(f"  {'Classical best':18s} ({b_name}):  {metric_label} = {b_val:.3f}{_ci_str(b_ci)}")
    if zs_val is not None:
        d_zs = delta(zs_val, b_val)
        sig  = "✓" if d_zs > 0 else "✗"
        print(f"  {'ZS best':18s} ({zs_name}):  {metric_label} = {zs_val:.3f}{_ci_str(zs_ci)}"
              f"   Δ={d_zs:+.3f} {sig}")
    if ft_val is not None:
        d_ft = delta(ft_val, b_val)
        sig  = "✓" if d_ft > 0 else "✗"
        print(f"  {'LP-FT best':18s} ({ft_name}):  {metric_label} = {ft_val:.3f}{_ci_str(ft_ci)}"
              f"   Δ={d_ft:+.3f} {sig}")

    # CI overlap test for ZS and LP-FT vs baseline
    if zs_val is not None:
        ov = _ci_overlap(zs_ci, b_ci)
        print(f"  ZS vs baseline CIs: {ov}")
    if ft_val is not None:
        ov = _ci_overlap(ft_ci, b_ci)
        print(f"  LP-FT vs baseline CIs: {ov}")

    # Wilcoxon if per-window arrays available
    win_errors = _load_window_errors(all_results, task)
    b_key  = f"baseline:{b_name}"
    ft_key = f"ft:{ft_name}"
    if b_key in win_errors and ft_key in win_errors:
        stat, p, d = _wilcoxon_paired(win_errors[b_key], win_errors[ft_key])
        p_adj  = min(p * N_TASKS, 1.0)
        sig    = "✓ SIGNIFICANT" if p_adj < ALPHA else "✗ not significant"
        d_lbl  = ("|d|<0.2 tiny" if abs(d) < 0.2 else
                  "|d|<0.5 small" if abs(d) < 0.5 else
                  "|d|<0.8 medium" if abs(d) < 0.8 else "|d|≥0.8 large")
        print(f"  Wilcoxon LP-FT vs best classical: p_adj={p_adj:.4f} d={d:+.3f} {sig} ({d_lbl})")
    else:
        print(f"  (Per-window errors not saved — Wilcoxon requires save_windows=True)")

    return b_val, zs_val, ft_val


def main():
    exp1      = _load(RESULTS_DIR / "exp1_results.json")
    exp2      = _load(RESULTS_DIR / "exp2_results.json")        # ZS
    exp3_lpft = _load(RESULTS_DIR / "exp3_lpft_results.json")   # LP-FT (frozen backbone + MLP)

    if not exp1:
        print("ERROR: exp1_results.json not found.")
        return

    all_results = _load(RESULTS_DIR / "all_results.json")

    print("=" * 78)
    print("Statistical Significance Analysis — LP-FT & ZS vs Classical Baselines")
    print(f"  32 active links  |  τ=0.10  |  Bonferroni α_adj = {ALPHA}/{N_TASKS} = {ALPHA_BONF:.4f}")
    print(f"  Comparisons: (1) ZS vs Classical  (2) LP-FT vs Classical")
    print("=" * 78)

    rows = []
    for task in TASKS:
        b_name, b_val, b_ci   = _best_baseline(exp1, task)
        zs_name, zs_val, zs_ci = _best_model(exp2, task) if exp2 else ("", None, None)
        ft_name, ft_val, ft_ci = _best_model(exp3_lpft, task) if exp3_lpft else ("", None, None)
        _print_task(task, b_name, b_val, b_ci,
                    zs_name, zs_val, zs_ci,
                    ft_name, ft_val, ft_ci,
                    all_results)
        rows.append((task, b_name, b_val, zs_name, zs_val, ft_name, ft_val))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY TABLE (for paper)")
    print("=" * 78)
    metric_names = {"t1_trajectory": "Skill", "t2_exceedance": "AUROC",
                    "t3_timing": "Skill", "t4_asym": "MAE"}
    print(f"\n{'Task':<16} {'Metric':>6}  {'Classical':>10}  {'ZS best':>10}  {'LP-FT best':>11}  {'Δ(LP-FT)':>9}")
    print("-" * 78)
    for task, b_name, b_val, zs_name, zs_val, ft_name, ft_val in rows:
        higher = task not in LOWER_IS_BETTER
        met    = metric_names[task]
        d_ft   = (ft_val - b_val) if (higher and ft_val is not None) else (
                  (b_val - ft_val) if ft_val is not None else None)
        zs_s   = f"{zs_val:.3f}" if zs_val is not None else "N/A"
        ft_s   = f"{ft_val:.3f}" if ft_val is not None else "N/A"
        d_s    = f"{d_ft:+.3f}" if d_ft is not None else "N/A"
        print(f"  {task:<14} {met:>6}  {b_val:>10.3f}  {zs_s:>10}  {ft_s:>11}  {d_s:>9}")

    print(f"""
Notes
-----
* T1 Skill_SN, T2 AUROC, T3 Skill: higher = better.  T4 MAE: lower = better.
* Δ(LP-FT) = LP-FT − classical for T1/T2/T3 (positive = LP-FT wins);
             = classical − LP-FT for T4 (positive = LP-FT wins).
* CI non-overlap (after Bonferroni) is sufficient for significance claim.
* To enable the Wilcoxon test: set save_windows=True in exp1_baseline.py and
  exp3_lp_ft.py, then re-run.
* EventRate ≈ 20% (τ=0.10, 32 active links, 60-min resolution).
""")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
