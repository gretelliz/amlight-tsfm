"""Experiment 3 (LP-FT) — Linear-Probe Fine-Tuning.

Linear-Probe Fine-Tuning (Kumar et al. 2022) bypasses Stage 1 domain
adaptation entirely: the backbone stays fully frozen and a lightweight
2-layer MLP head (64 hidden units, GELU, Dropout 0.1) is trained per task on
auto-labeled domain_adapt windows (paper §3.2). Because no backbone
parameter is ever updated, catastrophic forgetting is structurally
impossible — this is the row of Table 4 (paper §5.3) that avoids Failure
Mode 1 (catastrophic forgetting) but exposes Failure Mode 2 (task-selective
degradation when a task head's auto-label is less informative for that
task than for others).

Pipeline:
  1. Load the zero-shot model (no weight updates).
  2. Run inference on domain_adapt windows (up to 2,000) → extract features.
  3. Compute auto-labels for each task directly from telemetry.
  4. Train a 2-layer MLP per task on these (feature, label) pairs.
  5. Evaluate on the held-out test set using the frozen backbone + trained head.

Auto-labels (no human annotation):
  T1 (trajectory): MASE evaluated directly on the raw forecast — no head needed.
  T2 (exceedance): (traj[eval_start:].max > threshold).float()   binary
  T3 (peak timing): argmax(util) in eval sub-window               integer
  T4 (asymmetry):  mean(asym[eval_start:]) per link                float

Outputs: exp3_lpft_results.json
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data.loaders import (
    get_active_link_indices,
    load_all_splits,
    load_datasets,
)
from experiments.exp2_zeroshot import evaluate_model_zeroshot
from models import MODELS
from tasks.forecasting import (
    _compute_auroc, _brier_skill_score,
    compute_timing_mae, compute_timing_skill,
    make_trajectory_windows, make_peak_timing_windows, make_asym_windows,
    evaluate_trajectory_mase_tsfm, FrozenBackboneAdapter,
)


class _MLP(nn.Module):
    """Tiny 2-layer MLP task head: [in_features → 64 → out_features]."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_features),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp(X_tr: np.ndarray, y_tr: np.ndarray, task: str,
               lr: float = 1e-3, epochs: int = 80, batch: int = 64,
               val_frac: float = 0.1) -> _MLP:
    """Fit a small MLP head.  task in {'binary', 'regression'}."""
    n_val = max(1, int(len(X_tr) * val_frac))
    X_v, y_v = X_tr[-n_val:], y_tr[-n_val:]
    X_t, y_t = X_tr[:-n_val], y_tr[:-n_val]

    def _tensor(a, dtype):
        return torch.tensor(a, dtype=dtype)

    X_T = _tensor(X_t, torch.float32);  X_V = _tensor(X_v, torch.float32)
    if task == "binary":
        y_T = _tensor(y_t, torch.float32);  y_V = _tensor(y_v, torch.float32)
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        y_T = _tensor(y_t, torch.float32);  y_V = _tensor(y_v, torch.float32)
        loss_fn = nn.L1Loss()

    head   = _MLP(X_T.shape[1], y_T.shape[1] if y_T.ndim > 1 else 1)
    optim  = torch.optim.Adam(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_T, y_T), batch_size=batch, shuffle=True)

    best_val = float("inf");  best_state = None
    patience  = 15;  no_imp = 0

    for _ in range(epochs):
        head.train()
        for xb, yb in loader:
            optim.zero_grad()
            pred = head(xb).squeeze(-1) if yb.ndim == 1 else head(xb)
            loss_fn(pred, yb).backward()
            optim.step()

        head.eval()
        with torch.no_grad():
            pred_v = head(X_V).squeeze(-1) if y_V.ndim == 1 else head(X_V)
            val_l  = loss_fn(pred_v, y_V).item()
        if val_l < best_val:
            best_val = val_l;  best_state = {k: v.clone() for k, v in head.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def _predict_mlp(head: _MLP, X: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        out = head(torch.tensor(X, dtype=torch.float32))
    return out.numpy()


def run(cfg: dict, out_dir: Path, model_names: list[str] | None = None) -> dict:
    print("\n=== Experiment 3 (LP-FT): frozen backbone + MLP head on domain_adapt ===")
    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]

    splits    = load_all_splits(telemetry, datasets.get("split_info"))
    da_tel    = splits["domain_adapt"]
    test_tel  = splits["test"]

    weights_dir = Path(cfg["paths"]["weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    capacity   = cfg["data"]["capacity_gbps"]
    ctx_len    = cfg["model"]["context_len"]
    pred_len   = cfg["model"]["prediction_len"]
    eval_start = cfg["model"].get("eval_horizon_start", 0)
    eval_len   = pred_len - eval_start
    hours_ps   = cfg["data"].get("hours_per_step", 1.0)
    threshold  = cfg.get("tasks", {}).get("peak_threshold", 0.10)
    s_period   = cfg["baselines"].get("classical_seasonal_period", 24)

    active_idx = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    print("  Building domain-adapt windows for LP-FT …")
    N_DA = 2000  # cap to keep memory tractable
    ctx_da, traj_da = make_trajectory_windows(da_tel, ctx_len, pred_len, capacity, stride=4)
    ctx_da = ctx_da[:N_DA];  traj_da = traj_da[:N_DA]

    ctx_t3_da, timing_da = make_peak_timing_windows(da_tel, ctx_len, pred_len, capacity,
                                                      stride=4, eval_start=eval_start)
    ctx_t3_da = ctx_t3_da[:N_DA]
    timing_da = timing_da[:N_DA, :][:, active_idx]

    ctx_asym_da, tgt_asym_da = make_asym_windows(da_tel, ctx_len, pred_len,
                                                   stride=4, eval_start=eval_start)
    ctx_asym_da  = ctx_asym_da[:N_DA]
    tgt_asym_da  = tgt_asym_da[:N_DA, :][:, active_idx]

    # Auto-labels from telemetry
    y_exc_da = (traj_da[:, eval_start:, :]
                .max(axis=1)[:, active_idx] > threshold).astype(np.float32)

    # Test windows for evaluation
    N_TE = 200
    ctx_te,  traj_te  = make_trajectory_windows(test_tel, ctx_len, pred_len, capacity, stride=4)
    ctx_te   = ctx_te[:N_TE];  traj_te = traj_te[:N_TE]
    ctx_t3_te, timing_te = make_peak_timing_windows(test_tel, ctx_len, pred_len, capacity,
                                                      stride=4, eval_start=eval_start)
    ctx_t3_te = ctx_t3_te[:N_TE]
    timing_te_ev = timing_te[:N_TE, :][:, active_idx]
    ctx_asym_te, tgt_asym_te = make_asym_windows(test_tel, ctx_len, pred_len,
                                                   stride=4, eval_start=eval_start)
    ctx_asym_te  = ctx_asym_te[:N_TE]
    tgt_asym_te_ev = tgt_asym_te[:N_TE, :][:, active_idx]

    y_exc_te  = (traj_te[:, eval_start:, :].max(axis=1)[:, active_idx] > threshold).astype(np.float32)
    p_clim    = float(y_exc_te.mean())

    names     = model_names or ["chronos", "timesfm", "moirai", "patchtst", "ttm"]
    all_results: dict = {}

    for model_name in names:
        print(f"\n  {model_name} LP-FT …")
        gc.collect()

        model_cls = MODELS[model_name]
        extra_kw  = {}
        if model_name == "patchtst":
            extra_kw = {k: cfg["model"][k] for k in ("patch_len", "d_model", "n_heads", "n_layers", "dropout")}
        elif model_name in ("chronos", "moirai"):
            extra_kw = {"model_size": cfg.get("model_variant", {}).get(model_name, "large")}
        elif model_name == "ttm":
            extra_kw = {"model_size": cfg.get("model_variant", {}).get("ttm", "r2")}

        model = model_cls(context_len=ctx_len, prediction_len=pred_len, device=None, **extra_kw)
        model.load()
        # Backbone stays fully frozen throughout

        print(f"    [{model_name}] extracting predictions on domain_adapt …")

        # T1: MASE evaluated directly (no head)
        t1_res = evaluate_trajectory_mase_tsfm(
            model, ctx_da, traj_da, pred_len,
            link_indices=active_idx, max_windows=min(200, N_DA),
            seasonal_period=s_period, eval_start=eval_start,
        )
        print(f"    T1: MASE={t1_res.get('t1_mase', float('nan')):.3f}  "
              f"Skill_SN={t1_res.get('t1_skill_sn', float('nan')):.3f}")

        # --- Extract raw predictions for task heads ---
        adapter = FrozenBackboneAdapter(model, pred_len, link_indices=active_idx)

        # T2 input features: max exceedance score per active link (N, n_active)
        exc_da = adapter.get_exceedance_scores(ctx_da, threshold, eval_start=eval_start)
        exc_te = adapter.get_exceedance_scores(ctx_te, threshold, eval_start=eval_start)

        # T3 input: argmax(util) raw prediction per link
        raw_t3_da = adapter.get_raw_predictions(ctx_t3_da)[:, eval_start:, :].argmax(axis=1).astype(np.float32)
        raw_t3_te = adapter.get_raw_predictions(ctx_t3_te)[:, eval_start:, :].argmax(axis=1).astype(np.float32)

        # T4 input: mean(forecast_asym[eval_start:]) per active link
        # adapter already filters to n_active — do NOT re-index with active_idx
        raw_asym_da = adapter.get_raw_predictions(ctx_asym_da)[:, eval_start:, :].mean(axis=1)
        raw_asym_te = adapter.get_raw_predictions(ctx_asym_te)[:, eval_start:, :].mean(axis=1)

        # --- Train MLP heads ---
        print(f"    [{model_name}] training MLP heads (N={N_DA}) …")

        head_t2 = _train_mlp(exc_da,     y_exc_da,    "binary")
        head_t3 = _train_mlp(raw_t3_da,  timing_da,   "regression")
        head_t4 = _train_mlp(raw_asym_da, tgt_asym_da, "regression")

        torch.save(head_t2.state_dict(), weights_dir / f"{model_name}_lpft_head_t2.pt")
        torch.save(head_t3.state_dict(), weights_dir / f"{model_name}_lpft_head_t3.pt")
        torch.save(head_t4.state_dict(), weights_dir / f"{model_name}_lpft_head_t4.pt")
        print(f"    [{model_name}] MLP heads saved to {weights_dir}")

        # --- Evaluate on test set ---
        print(f"    [{model_name}] evaluating on test …")

        # T2: binary → AUROC + BSS
        p_hat_te = torch.sigmoid(torch.tensor(
            _predict_mlp(head_t2, exc_te))).numpy()  # (N_TE, n_active)
        t2_auroc = _compute_auroc(y_exc_te.ravel(), p_hat_te.ravel())
        t2_bss   = _brier_skill_score(y_exc_te.ravel(), p_hat_te.ravel(), p_clim)
        t2_rate  = float(y_exc_te.mean())
        print(f"    T2: AUROC={t2_auroc:.3f}  BSS={t2_bss:.3f}  EventRate={t2_rate:.2%}")

        # T3: regression → MAE(h) + Skill
        tim_pred_cal = np.clip(_predict_mlp(head_t3, raw_t3_te), 0, eval_len - 1)
        t3_mae = compute_timing_mae(timing_te_ev, tim_pred_cal, hours_ps)
        t3_sk  = compute_timing_skill(t3_mae, eval_len, hours_ps)
        print(f"    T3: MAE={t3_mae:.2f}h  Skill={t3_sk:.3f}")

        # T4: regression → MAE
        asym_pred_cal = np.clip(_predict_mlp(head_t4, raw_asym_te), -1, 1)
        t4_mae = float(np.abs(asym_pred_cal - tgt_asym_te_ev).mean())
        print(f"    T4: MAE={t4_mae:.4f}")

        model_results = {
            "t1_trajectory": t1_res,
            "t2_exceedance": {
                "t2_auroc":      t2_auroc,
                "t2_bss":        t2_bss,
                "t2_event_rate": t2_rate,
            },
            "t3_timing": {
                "mae_h":    t3_mae,
                "t3_skill": t3_sk,
            },
            "t4_asym": {
                "t4_mae": t4_mae,
            },
            "meta": {
                "method":   "lp_ft_mlp",
                "n_da_windows": N_DA,
                "backbone": "frozen_zeroshot",
            },
        }
        all_results[model_name] = model_results
        del model;  gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    cache  = out_dir / "exp3_lpft_results.json"
    merged = json.loads(cache.read_text()) if cache.exists() else {}
    merged.update(all_results)
    cache.write_text(json.dumps(merged, indent=2))
    print(f"  Results saved to {cache}")
    return merged


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
