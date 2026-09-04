"""Experiment 3 (reduced-LR ablation) — TimesFM Full DA at lower learning rates.

exp3_full_da.py uses lr = base_lr * 0.3 = 3e-5 for TimesFM and observes
catastrophic forgetting (paper §5.3, Failure Mode 1: T1 SkillSN +0.387 → -2.59,
T2 AUROC 0.899 → 0.750). This ablation asks: is the forgetting LR-dependent?

  lr = base_lr * 0.10 = 1e-5  (×3  reduction)
  lr = base_lr * 0.03 = 3e-6  (×10 reduction)

The paper reports that TimesFM forgetting is learning-rate independent: even
at 10x lower LR, catastrophic T1 collapse is still observed.

Results stored as:
  timesfm_da_lr1e5   (1e-5 run)
  timesfm_da_lr3e6   (3e-6 run)

Outputs: exp3_reducedlr_results.json, exp3_reducedlr_histories.json
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.loaders import (
    build_window_datasets,
    get_active_link_indices,
    load_all_splits,
    load_datasets,
)
from experiments.exp2_zeroshot import evaluate_model_zeroshot
from models import MODELS

_LR_VARIANTS: list[tuple[str, float]] = [
    ("timesfm_da_lr1e5", 1e-5),
    ("timesfm_da_lr3e6", 3e-6),
]


def run(cfg: dict, out_dir: Path) -> dict:
    print("\n=== Experiment 3 (reduced-LR ablation): TimesFM Full DA ===")
    print(f"  LR variants: {[f'{name}={lr:.0e}' for name, lr in _LR_VARIANTS]}")

    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]

    splits   = load_all_splits(telemetry, datasets.get("split_info"))
    da_tel   = splits["domain_adapt"]
    test_tel = splits["test"]

    n_da     = len(da_tel)
    n_da_val = max(int(n_da * 0.15),
                   cfg["model"]["context_len"] + cfg["model"]["prediction_len"] + 50)
    da_train_tel = da_tel.iloc[: n_da - n_da_val]
    da_val_tel   = da_tel.iloc[n_da - n_da_val :]

    active_idx = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    da_thresh    = cfg["data"].get("da_link_threshold",
                                   cfg["data"].get("active_link_threshold", 0.005))
    da_idx       = get_active_link_indices(da_train_tel, da_thresh)
    active_cols  = da_train_tel.columns[da_idx]
    da_train_act = da_train_tel[active_cols]
    da_val_act   = da_val_tel[active_cols]

    ft_stride = cfg["finetune"].get("stride_da", 4)
    train_ds, val_ds, _, _ = build_window_datasets(
        da_train_act, da_val_act, da_val_act,
        context_len=cfg["model"]["context_len"],
        prediction_len=cfg["model"]["prediction_len"],
        capacity_gbps=cfg["data"]["capacity_gbps"],
        masked=True, stride=ft_stride,
    )
    batch_size = max(8, cfg["finetune"]["batch_size"] // 4)

    weights_dir = Path(cfg["paths"]["weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    all_results:   dict = {}
    all_histories: dict = {}

    for run_key, lr in _LR_VARIANTS:
        print(f"\n  [{run_key}] TimesFM Full DA, lr={lr:.0e} …")
        gc.collect()
        torch.cuda.empty_cache()

        model = MODELS["timesfm"](
            context_len=cfg["model"]["context_len"],
            prediction_len=cfg["model"]["prediction_len"],
            device=None,
        )
        model.load()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        history = model.finetune(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["finetune"]["epochs"],
            lr=lr,
            weight_decay=cfg["finetune"]["weight_decay"],
            patience=cfg["finetune"]["patience"],
            weights_dir=weights_dir,
        )
        all_histories[run_key] = history
        model.save_weights(weights_dir / f"{run_key}.pt")

        model_results = evaluate_model_zeroshot(
            model, cfg, test_tel, active_idx,
            eval_start=cfg["model"].get("eval_horizon_start", 0),
            model_name="timesfm",
        )
        all_results[run_key] = model_results

        # Print summary
        t1  = model_results.get("t1_trajectory", {})
        t2  = model_results.get("t2_exceedance", {})
        t3  = model_results.get("t3_timing", {})
        t4  = model_results.get("t4_asym", {})
        print(f"    T1 Skill_SN = {t1.get('t1_skill_sn', float('nan')):.3f}")
        print(f"    T2 AUROC    = {t2.get('t2_auroc',    float('nan')):.3f}")
        print(f"    T3 Skill    = {t3.get('t3_skill',    float('nan')):.3f}")
        print(f"    T4 MAE      = {t4.get('t4_mae',      float('nan')):.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in [("exp3_reducedlr_results.json",   all_results),
                        ("exp3_reducedlr_histories.json", all_histories)]:
        cache  = out_dir / fname
        merged = json.loads(cache.read_text()) if cache.exists() else {}
        merged.update(data)
        cache.write_text(json.dumps(merged, indent=2))

    print(f"\n  Results saved → {out_dir}/exp3_reducedlr_results.json")
    print("\n  LR ablation summary:")
    print(f"  {'Run':<25} {'T1 Sk_SN':>10} {'T2 AUROC':>10} {'T3 Skill':>10} {'T4 MAE':>8}")
    for key, res in all_results.items():
        t1 = res.get("t1_trajectory", {}).get("t1_skill_sn", float("nan"))
        t2 = res.get("t2_exceedance", {}).get("t2_auroc",    float("nan"))
        t3 = res.get("t3_timing",     {}).get("t3_skill",    float("nan"))
        t4 = res.get("t4_asym",       {}).get("t4_mae",      float("nan"))
        print(f"  {key:<25} {t1:>10.3f} {t2:>10.3f} {t3:>10.3f} {t4:>8.4f}")

    return all_results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
