"""Experiment 4 — Few-shot learning curves (ZS+MLP@n, Stage 2 task adaptation).

Freezes the zero-shot backbone (no Stage 1 domain adaptation) and fits a
fresh 2-layer MLP head per task on N auto-labeled windows drawn from the
train split, N in {5, 10, 25, 50, 100, 500} (paper §3.2, §5.4). Answers RQ3:
how few labeled examples suffice for a task head to reach operationally
useful performance?

T1 (trajectory) is excluded from this sweep: the backbone already produces
the 48-step trajectory natively, so there is no task head to train and no
label-efficiency question to ask for it (see paper §3.2). T1 numbers are
those reported by Experiment 2 (zero-shot).

Each task head uses the loss function that matches its target — binary
cross-entropy for T2, MSE for T3/T4 — directly optimizing the right
objective for that task (unlike LP-FT's single shared timing auto-label;
see exp3_lp_ft.py and the paper's Failure Mode 2 discussion).

Results are merged into exp4_results.json under keys:
  t2_exceedance_mlp, t3_timing_mlp, t4_asym_mlp
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.loaders import (
    get_active_link_indices,
    load_all_splits,
    load_datasets,
)
from experiments.exp3_lp_ft import _MLP, _train_mlp
from models import MODELS
from tasks.forecasting import (
    FrozenBackboneAdapter,
    _compute_auroc,
    _brier_skill_score,
    compute_timing_mae,
    compute_timing_skill,
    make_trajectory_windows,
    make_asym_windows,
)


def _apply_mlp(head: _MLP, X: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(torch.tensor(X, dtype=torch.float32)).numpy()


def _get_asym_pred_mean(adapter: FrozenBackboneAdapter, ctx_asym: np.ndarray,
                        eval_start: int) -> np.ndarray:
    raw = adapter.get_raw_predictions(ctx_asym)
    return raw[:, eval_start:, :].mean(axis=1)


def run(cfg: dict, out_dir: Path, model_names: list[str] | None = None) -> dict:
    print("\n=== Experiment 4: Few-shot Learning Curves (ZS+MLP@n) ===")
    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]

    splits    = load_all_splits(telemetry, datasets.get("split_info"))
    train_tel = pd.concat([splits["train"], splits["val"]])
    test_tel  = splits["test"]

    capacity    = cfg["data"]["capacity_gbps"]
    context_len = cfg["model"]["context_len"]
    pred_len    = cfg["model"]["prediction_len"]
    eval_start  = cfg["model"].get("eval_horizon_start", 0)
    eval_len    = pred_len - eval_start
    hours_ps    = cfg["data"].get("hours_per_step", 1.0)
    threshold   = cfg.get("tasks", {}).get("peak_threshold", 0.10)

    active_idx = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    print("  Building context windows …")
    ctx_tr, traj_tr = make_trajectory_windows(train_tel, context_len, pred_len, capacity, stride=4)
    ctx_te, traj_te = make_trajectory_windows(test_tel,  context_len, pred_len, capacity, stride=4)

    y_exc_tr = (traj_tr[:, eval_start:, :].max(axis=1)[:, active_idx] > threshold).astype(np.float32)
    y_exc_te = (traj_te[:, eval_start:, :].max(axis=1)[:, active_idx] > threshold).astype(np.float32)

    timing_tr = traj_tr[:, eval_start:, :].argmax(axis=1).astype("float32")[:, active_idx]
    timing_te = traj_te[:, eval_start:, :].argmax(axis=1).astype("float32")[:, active_idx]

    ctx_asym_tr, tgt_asym_tr = make_asym_windows(train_tel, context_len, pred_len,
                                                   stride=4, eval_start=eval_start)
    ctx_asym_te, tgt_asym_te = make_asym_windows(test_tel,  context_len, pred_len,
                                                   stride=4, eval_start=eval_start)
    tgt_asym_tr = tgt_asym_tr[:, active_idx]
    tgt_asym_te = tgt_asym_te[:, active_idx]

    N_TRAIN = min(2000, len(ctx_tr))
    ctx_tr      = ctx_tr[:N_TRAIN];      traj_tr     = traj_tr[:N_TRAIN]
    y_exc_tr    = y_exc_tr[:N_TRAIN];    timing_tr   = timing_tr[:N_TRAIN]
    ctx_asym_tr = ctx_asym_tr[:N_TRAIN]; tgt_asym_tr = tgt_asym_tr[:N_TRAIN]
    print(f"  Training windows: {N_TRAIN}  T2 τ={threshold} event rate={float(y_exc_tr.mean()):.1%}")

    N_P          = 200
    y_exc_te_ev  = y_exc_te[:N_P]
    timing_te_ev = timing_te[:N_P]
    p_clim_te    = float(y_exc_te_ev.mean())

    n_labeled_list = [n for n in cfg["fewshot"]["n_labeled"] if n <= N_TRAIN]
    n_trials       = cfg["fewshot"]["n_trials"]
    rng            = np.random.default_rng(cfg.get("seed", 42))

    # T1 excluded: the ZS backbone already produces the trajectory natively.
    # PatchTST excluded: Stage 2 uses the pretrained ZS backbone as a fixed
    # feature extractor, and PatchTST has no pre-trained checkpoint.
    names = model_names or ["chronos", "timesfm", "moirai", "ttm"]
    all_curves: dict[str, dict] = {}

    for model_name in names:
        print(f"\n  [{model_name}] loading ZS model …")
        gc.collect()
        torch.cuda.empty_cache()

        model_cls = MODELS[model_name]
        extra_kw  = {}
        if model_name in ("chronos", "moirai"):
            extra_kw = {"model_size": cfg.get("model_variant", {}).get(model_name, "large")}
        elif model_name == "ttm":
            extra_kw = {"model_size": cfg.get("model_variant", {}).get("ttm", "r2")}

        model = model_cls(context_len=context_len, prediction_len=pred_len, device=None, **extra_kw)
        model.load()

        adapter_zs = FrozenBackboneAdapter(model, pred_len, link_indices=active_idx)
        print(f"    [{model_name}] ZS predictions …")
        exc_tr_zs      = adapter_zs.get_exceedance_scores(ctx_tr, threshold, eval_start=eval_start)
        exc_te_zs      = adapter_zs.get_exceedance_scores(ctx_te[:N_P], threshold, eval_start=eval_start)
        raw_t_tr_zs    = adapter_zs.get_raw_predictions(ctx_tr)[:, eval_start:, :].argmax(axis=1).astype("float32")
        raw_t_te_zs    = adapter_zs.get_raw_predictions(ctx_te[:N_P])[:, eval_start:, :].argmax(axis=1).astype("float32")
        asym_pred_tr_zs = _get_asym_pred_mean(adapter_zs, ctx_asym_tr, eval_start)
        asym_pred_te_zs = _get_asym_pred_mean(adapter_zs, ctx_asym_te[:N_P], eval_start)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        task_curves: dict[str, list] = {
            "t2_exceedance_mlp": [],
            "t3_timing_mlp":     [],
            "t4_asym_mlp":       [],
        }

        for n_lab in n_labeled_list:
            t2_auroc_mlp, t2_bss_mlp = [], []
            mae_t3_mlp, mae_t4_mlp   = [], []
            mlp_epochs = min(80, max(20, n_lab * 2))

            for _ in range(n_trials):
                idx      = rng.choice(N_TRAIN, min(n_lab, N_TRAIN), replace=False)
                idx_asym = rng.choice(len(ctx_asym_tr), min(n_lab, len(ctx_asym_tr)), replace=False)

                # T2 — binary MLP, sigmoid output as P(exceedance)
                mlp_t2 = _train_mlp(exc_tr_zs[idx], y_exc_tr[idx], "binary", epochs=mlp_epochs)
                mlp_t2.eval()
                with torch.no_grad():
                    logits_te = mlp_t2(torch.tensor(exc_te_zs, dtype=torch.float32)).numpy()
                p_hat = torch.sigmoid(torch.tensor(logits_te)).numpy()
                t2_auroc_mlp.append(_compute_auroc(y_exc_te_ev.ravel(), p_hat.ravel()))
                t2_bss_mlp.append(_brier_skill_score(y_exc_te_ev.ravel(), p_hat.ravel(), p_clim_te))
                del mlp_t2

                # T3 — regression MLP on peak timing
                mlp_t3 = _train_mlp(raw_t_tr_zs[idx], timing_tr[idx], "regression", epochs=mlp_epochs)
                pred_t3 = np.clip(_apply_mlp(mlp_t3, raw_t_te_zs), 0, eval_len - 1)
                mae_t3_mlp.append(compute_timing_mae(timing_te_ev, pred_t3, hours_ps))
                del mlp_t3

                # T4 — regression MLP on asymmetry
                mlp_t4 = _train_mlp(asym_pred_tr_zs[idx_asym], tgt_asym_tr[idx_asym], "regression", epochs=mlp_epochs)
                pred_t4 = np.clip(_apply_mlp(mlp_t4, asym_pred_te_zs), -1.0, 1.0)
                mae_t4_mlp.append(float(np.abs(pred_t4 - tgt_asym_te[:N_P]).mean()))
                del mlp_t4

            t3_sk = compute_timing_skill(float(np.mean(mae_t3_mlp)), eval_len, hours_ps)
            print(
                f"    N={n_lab:4d}  "
                f"T2 AUROC={np.mean(t2_auroc_mlp):.3f}  "
                f"T3 Skill={t3_sk:.3f}  "
                f"T4 MAE={np.mean(mae_t4_mlp):.4f}"
            )

            task_curves["t2_exceedance_mlp"].append({
                "n": n_lab,
                "auroc_mean": float(np.mean(t2_auroc_mlp)), "auroc_std": float(np.std(t2_auroc_mlp)),
                "bss_mean":   float(np.mean(t2_bss_mlp)),   "bss_std":   float(np.std(t2_bss_mlp)),
            })
            mae_mean_t3 = float(np.mean(mae_t3_mlp))
            task_curves["t3_timing_mlp"].append({
                "n": n_lab,
                "mae_mean":   mae_mean_t3,
                "mae_std":    float(np.std(mae_t3_mlp)),
                "skill_mean": compute_timing_skill(mae_mean_t3, eval_len, hours_ps),
            })
            task_curves["t4_asym_mlp"].append({
                "n": n_lab,
                "mae_mean": float(np.mean(mae_t4_mlp)),
                "mae_std":  float(np.std(mae_t4_mlp)),
            })

        all_curves[model_name] = task_curves

    # Merge into exp4_results.json (only adds _mlp keys, preserves everything else)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "exp4_results.json"
    merged = json.loads(cache.read_text()) if cache.exists() else {}
    for model_name, curves in all_curves.items():
        if model_name not in merged:
            merged[model_name] = {}
        merged[model_name].update(curves)
    cache.write_text(json.dumps(merged, indent=2))
    print(f"\n  Results merged → {cache}")
    return merged


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
