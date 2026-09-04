"""Experiment 2 — Zero-shot TSFM evaluation.

Applies pre-trained TSFMs to AmLight telemetry with no domain fine-tuning.
Measures performance on all four tasks and compares to the classical
baselines from Experiment 1 (paper §5.2, Table 3).
Evaluation is restricted to active links (mean util ≥ 0.5%).

PatchTST has no pre-trained checkpoint (trained from scratch in Experiment 3)
and is therefore excluded from zero-shot evaluation.

Tasks:
    T1 — Trajectory Forecasting     (MASE / Skill_SN; CRPS for probabilistic models)
    T2 — Peak Exceedance Detection  (AUROC + BSS)
    T3 — Peak Timing Prediction     (MAE + Skill)
    T4 — Asymmetry Forecasting      (MAE + Skill vs persistence and seasonal naive)
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from data.loaders import load_all_splits, load_datasets, get_active_link_indices
from models import MODELS, TSFMBase
from tasks.forecasting import (
    evaluate_exceedance_tsfm,
    evaluate_timing_tsfm,
    evaluate_trajectory_mase_tsfm,
    evaluate_trajectory_crps_tsfm,
    evaluate_asym_tsfm,
    make_trajectory_windows,
    make_peak_timing_windows,
    make_asym_windows,
)


_PROBABILISTIC_MODELS = {"chronos", "moirai"}


def evaluate_model_zeroshot(
    model: TSFMBase,
    cfg: dict,
    test_tel,
    active_idx: list[int] | None = None,
    eval_start: int = 0,
    model_name: str = "",
) -> dict[str, dict]:
    capacity = cfg["data"]["capacity_gbps"]
    ctx_len  = cfg["model"]["context_len"]
    pred_len = cfg["model"]["prediction_len"]
    hours_ps = cfg["data"].get("hours_per_step", 1.0)
    s_period = cfg["baselines"].get("classical_seasonal_period", 24)

    results = {}

    # T1 — Trajectory MASE (all models) + CRPS (probabilistic models only)
    print(f"    [{model.name}] T1 trajectory MASE …")
    ctx_t1, traj_t1 = make_trajectory_windows(test_tel, ctx_len, pred_len, capacity, stride=4)
    results["t1_trajectory"] = evaluate_trajectory_mase_tsfm(
        model, ctx_t1, traj_t1, pred_len,
        link_indices=active_idx, max_windows=200, seasonal_period=s_period, eval_start=eval_start,
    )
    if model_name in _PROBABILISTIC_MODELS:
        print(f"    [{model.name}] T1b trajectory CRPS …")
        crps_res = evaluate_trajectory_crps_tsfm(
            model, ctx_t1, traj_t1, pred_len,
            link_indices=active_idx, max_windows=100, seasonal_period=s_period, eval_start=eval_start,
        )
        results["t1_trajectory"].update({f"crps_{k}": v for k, v in crps_res.items()})

    # T2 — Peak Utilization Exceedance Detection (AUROC + BSS)
    threshold = cfg.get("tasks", {}).get("peak_threshold", 0.10)
    print(f"    [{model.name}] T2 peak exceedance (τ={threshold}, eval_start={eval_start}) …")
    ctx_t2, traj_t2 = make_trajectory_windows(test_tel, ctx_len, pred_len, capacity, stride=4)
    results["t2_exceedance"] = evaluate_exceedance_tsfm(
        model, ctx_t2, traj_t2, pred_len,
        threshold=threshold, link_indices=active_idx, max_windows=200, eval_start=eval_start,
    )

    # T3 — Peak Timing Prediction (MAE + Skill)
    print(f"    [{model.name}] T3 peak timing (eval_start={eval_start}) …")
    ctx_t3, timing_t3 = make_peak_timing_windows(test_tel, ctx_len, pred_len, capacity,
                                                  stride=4, eval_start=eval_start)
    results["t3_timing"] = evaluate_timing_tsfm(
        model, ctx_t3, timing_t3, pred_len,
        link_indices=active_idx, hours_per_step=hours_ps, max_windows=200, eval_start=eval_start,
    )

    # T4 — Asymmetry Forecasting
    print(f"    [{model.name}] T4 asymmetry forecasting …")
    ctx_asym, tgt_asym = make_asym_windows(test_tel, ctx_len, pred_len, stride=4, eval_start=eval_start)
    results["t4_asym"] = evaluate_asym_tsfm(
        model, ctx_asym, tgt_asym, pred_len,
        link_indices=active_idx, max_windows=200, seasonal_period=s_period, eval_start=eval_start,
    )

    return results


def run(cfg: dict, out_dir: Path, model_names: list[str] | None = None) -> dict:
    print("\n=== Experiment 2: Zero-shot TSFM Evaluation ===")
    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]

    splits    = load_all_splits(telemetry, datasets.get("split_info"))
    test_tel  = splits["test"]

    active_idx = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    all_results = {}
    names = model_names or ["chronos", "timesfm", "moirai", "ttm"]

    for model_name in names:
        print(f"\n  Loading {model_name} …")
        model_cls = MODELS[model_name]
        size_kw = {}
        if model_name in ("chronos", "moirai"):
            size_kw["model_size"] = cfg.get("model_variant", {}).get(model_name, "large")
        if model_name == "patchtst":
            size_kw = {k: cfg["model"][k] for k in ("patch_len", "d_model", "n_heads", "n_layers", "dropout")}
        if model_name == "ttm":
            size_kw["model_size"] = cfg.get("model_variant", {}).get("ttm", "r2")
        model = model_cls(
            context_len=cfg["model"]["context_len"],
            prediction_len=cfg["model"]["prediction_len"],
            device=None,
            **size_kw,
        )
        model.load()
        model_results = evaluate_model_zeroshot(
            model, cfg, test_tel, active_idx,
            eval_start=cfg["model"].get("eval_horizon_start", 0),
            model_name=model_name,
        )
        all_results[model_name] = model_results

        for task, m in model_results.items():
            if task == "t1_trajectory":
                mase = m.get("t1_mase", float("nan"))
                sk   = m.get("t1_skill_sn", float("nan"))
                crps = m.get("crps_t1_crps", float("nan"))
                print(f"    {model_name} / {task}: MASE={mase:.3f}  Skill_SN={sk:.3f}  "
                      f"CRPS={crps:.4f}" if crps == crps else
                      f"    {model_name} / {task}: MASE={mase:.3f}  Skill_SN={sk:.3f}")
            elif task == "t2_exceedance":
                ci = m.get("t2_auroc_ci", [float("nan"), float("nan")])
                print(f"    {model_name} / {task}: "
                      f"AUROC={m.get('t2_auroc', float('nan')):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
                      f"BSS={m.get('t2_bss', float('nan')):.3f}  "
                      f"EventRate={m.get('t2_event_rate', float('nan')):.2%}")
            elif task == "t4_asym":
                mae  = m.get("t4_mae", float("nan"))
                sk_p = m.get("t4_skill_persist", float("nan"))
                sk_s = m.get("t4_skill_sn", float("nan"))
                print(f"    {model_name} / {task}: MAE={mae:.4f}  "
                      f"Skill_P={sk_p:.3f}  Skill_SN={sk_s:.3f}")
            else:  # t3_timing
                mae = m.get("mae_h", float("nan"))
                sk  = m.get("t3_skill", float("nan"))
                skp = m.get("t3_skill_persist", float("nan"))
                ci  = m.get("mae_ci", [float("nan"), float("nan")])
                print(f"    {model_name} / {task}: Skill(rand)={sk:.3f}  "
                      f"Skill(persist)={skp:.3f}  MAE={mae:.2f}h [{ci[0]:.2f}, {ci[1]:.2f}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "exp2_results.json"
    merged = json.loads(cache.read_text()) if cache.exists() else {}
    merged.update(all_results)
    cache.write_text(json.dumps(merged, indent=2))
    print(f"  Results saved to {cache}")
    return merged


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
