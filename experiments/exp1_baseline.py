"""Experiment 1 — Rule-based and classical statistical baselines.

Measures performance on all four operational tasks at 60-min resolution
before any TSFM is introduced. Sets the performance floor every model must
exceed (paper §5.1, Table 2).

Evaluation is restricted to active links (mean util ≥ 0.5%) to exclude idle links.

Tasks:
    T1 — Trajectory Forecasting   (MASE vs seasonal naive; Skill_SN = 1 - MASE)
    T2 — Peak Exceedance Detection (AUROC + BSS; τ=peak_threshold from config)
    T3 — Peak Timing Prediction   (MAE(h) + Skill vs random 8 h)
    T4 — Asymmetry Forecasting    (MAE + Skill vs persistence and seasonal naive)

Baselines per task:
    B1 — persistence exceedance: score = max(last eval_len context steps)
    B3 — persistence timing:     argmax of last eval_len context steps
    B4 — persistence asym:       last observed asym value
    Classical: ARIMA(1,0,1), SARIMA(1,0,1)(1,0,0,24), Prophet, Holt-Winters
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from data.loaders import load_all_splits, load_datasets, get_active_link_indices
from tasks.baselines_classical import (
    evaluate_classical_exceedance,
    evaluate_classical_timing,
    evaluate_classical_trajectory_mase,
    evaluate_classical_asym,
)
from tasks.forecasting import (
    evaluate_exceedance_persistence,
    evaluate_timing_persistence,
    make_trajectory_windows,
    make_peak_timing_windows,
    make_asym_windows,
    evaluate_asym_persistence,
)


def _run_classical(fn, *args, label: str, metric: str = "t3", **kwargs) -> dict | None:
    try:
        result = fn(*args, **kwargs)
        if metric == "t2":
            ci = result.get("t2_auroc_ci", [float("nan"), float("nan")])
            print(f"    {label:30s}  AUROC={result.get('t2_auroc', float('nan')):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
                  f"BSS={result.get('t2_bss', float('nan')):.3f}")
        else:  # t3 timing
            mae = result.get("mae_h", float("nan"))
            sk  = result.get("t3_skill", float("nan"))
            print(f"    {label:30s}  MAE={mae:.2f}h  Skill={sk:.3f}")
        return result
    except ImportError as exc:
        print(f"    {label:30s}  SKIPPED ({exc})")
        return None


def run(cfg: dict, out_dir: Path) -> dict:
    print("\n=== Experiment 1: Rule-based and Classical Baselines ===")
    datasets = load_datasets(cfg["data"])

    telemetry = datasets["telemetry"]
    splits    = load_all_splits(telemetry, datasets.get("split_info"))
    test_tel  = splits["test"]

    context_len    = cfg["model"]["context_len"]
    pred_len       = cfg["model"]["prediction_len"]
    eval_start     = cfg["model"].get("eval_horizon_start", 0)
    capacity       = cfg["data"]["capacity_gbps"]
    s_period       = cfg["baselines"].get("classical_seasonal_period", 24)
    hours_ps       = cfg["data"].get("hours_per_step", 1.0)

    # Active links computed on full telemetry for consistency across all splits
    active_idx  = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    N_EVAL      = 500
    N_CLASSICAL = cfg["baselines"].get("classical_max_windows", 20)
    results     = {}

    # ------------------------------------------------------------------
    # T2 — Peak Utilization Exceedance Detection (AUROC + BSS)
    # ------------------------------------------------------------------
    threshold = cfg.get("tasks", {}).get("peak_threshold", 0.10)
    print(f"  T2: Peak Exceedance Detection (τ={threshold}) …")
    ctx_te, traj_te = make_trajectory_windows(
        test_tel, context_len, pred_len, capacity, stride=4
    )
    ctx_te   = ctx_te[:N_EVAL]
    traj_te  = traj_te[:N_EVAL]

    m_b1 = evaluate_exceedance_persistence(
        ctx_te, traj_te, pred_len, threshold=threshold,
        link_indices=active_idx, max_windows=N_EVAL, eval_start=eval_start,
    )
    print(f"    B1 persistence  AUROC={m_b1['t2_auroc']:.3f}  BSS={m_b1['t2_bss']:.3f}  "
          f"EventRate={m_b1['t2_event_rate']:.2%}")

    t2_classical = {}
    for method in ("arima", "sarima", "holtwinters", "prophet"):
        res = _run_classical(
            evaluate_classical_exceedance,
            ctx_te, traj_te, pred_len,
            method=method, max_windows=N_CLASSICAL, seasonal_period=s_period,
            threshold=threshold, link_indices=active_idx, eval_start=eval_start,
            label=f"T2 {method:12s}", metric="t2",
        )
        if res is not None:
            t2_classical[method] = res

    results["t2_exceedance"] = {
        "b1_persistence": m_b1,
        **{f"classical_{k}": v for k, v in t2_classical.items()},
    }

    # ------------------------------------------------------------------
    # T3 — Peak Timing Prediction
    # ------------------------------------------------------------------
    print("  T3: Peak Timing Prediction …")
    ctx_t3_te, timing_te = make_peak_timing_windows(
        test_tel, context_len, pred_len, capacity, stride=4, eval_start=eval_start
    )
    ctx_t3_te = ctx_t3_te[:N_EVAL]
    timing_te = timing_te[:N_EVAL]

    m_b3 = evaluate_timing_persistence(
        ctx_t3_te, timing_te, pred_len,
        link_indices=active_idx, hours_per_step=hours_ps,
        max_windows=N_EVAL, eval_start=eval_start,
    )
    print(f"    B3 persist. timing  Skill={m_b3['t3_skill']:.3f}  MAE={m_b3['mae_h']:.2f}h  CI={m_b3['mae_ci']}")

    t3_classical = {}
    for method in ("arima", "sarima", "holtwinters", "prophet"):
        res = _run_classical(
            evaluate_classical_timing,
            ctx_t3_te, timing_te, pred_len,
            method=method, max_windows=N_CLASSICAL, seasonal_period=s_period,
            link_indices=active_idx, hours_per_step=hours_ps, eval_start=eval_start,
            label=f"T3 {method:12s}", metric="t3",
        )
        if res is not None:
            t3_classical[method] = res

    results["t3_timing"] = {
        "b3_persistence_timing": m_b3,
        **{f"classical_{k}": v for k, v in t3_classical.items()},
    }

    # ------------------------------------------------------------------
    # T1 — Trajectory Forecasting (MASE vs seasonal naive)
    # ------------------------------------------------------------------
    print("  T1: Trajectory Forecasting (MASE) …")
    t1_classical = {}
    for method in ("arima", "sarima", "holtwinters", "prophet"):
        try:
            res = evaluate_classical_trajectory_mase(
                ctx_te, traj_te, pred_len,
                method=method, max_windows=N_CLASSICAL, seasonal_period=s_period,
                link_indices=active_idx, eval_start=eval_start,
            )
            mase = res.get("t1_mase", float("nan"))
            sk   = res.get("t1_skill_sn", float("nan"))
            print(f"    T1 {method:12s}  MASE={mase:.3f}  Skill_SN={sk:.3f}")
            t1_classical[method] = res
        except ImportError as exc:
            print(f"    T1 {method:12s}  SKIPPED ({exc})")

    results["t1_trajectory"] = {**{f"classical_{k}": v for k, v in t1_classical.items()}}

    # ------------------------------------------------------------------
    # T4 — Asymmetry Forecasting
    # ------------------------------------------------------------------
    print("  T4: Asymmetry Forecasting …")
    ctx_asym_te, tgt_asym_te = make_asym_windows(
        test_tel, context_len, pred_len, stride=4, eval_start=eval_start
    )
    ctx_asym_te = ctx_asym_te[:N_EVAL]
    tgt_asym_te = tgt_asym_te[:N_EVAL]

    m_b4 = evaluate_asym_persistence(
        ctx_asym_te, tgt_asym_te,
        link_indices=active_idx, max_windows=N_EVAL,
    )
    print(f"    B4 persist. asym  MAE={m_b4['t4_mae']:.4f}  Skill_SN={m_b4.get('t4_skill_sn', float('nan')):.3f}")

    t4_classical = {}
    for method in ("arima", "sarima", "holtwinters", "prophet"):
        try:
            res = evaluate_classical_asym(
                ctx_asym_te, tgt_asym_te, pred_len,
                method=method, max_windows=N_CLASSICAL, seasonal_period=s_period,
                link_indices=active_idx, eval_start=eval_start,
            )
            mae  = res.get("t4_mae", float("nan"))
            sk_p = res.get("t4_skill_persist", float("nan"))
            sk_s = res.get("t4_skill_sn", float("nan"))
            print(f"    T4 {method:12s}  MAE={mae:.4f}  Skill_P={sk_p:.3f}  Skill_SN={sk_s:.3f}")
            t4_classical[method] = res
        except ImportError as exc:
            print(f"    T4 {method:12s}  SKIPPED ({exc})")

    results["t4_asym"] = {
        "b4_persistence": m_b4,
        **{f"classical_{k}": v for k, v in t4_classical.items()},
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp1_results.json").write_text(json.dumps(results, indent=2))
    print(f"  Results saved to {out_dir}/exp1_results.json")
    return results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
