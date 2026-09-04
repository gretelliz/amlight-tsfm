"""Experiment 3 (Full DA) — Stage 1 domain adaptation on AmLight telemetry.

Full-parameter domain adaptation: every backbone weight is updated using the
model's native self-supervised pre-training objective on unlabeled AmLight
telemetry (no task labels). This is the "Full DA (weight-update)" row of
Table 4 (paper §5.3) and is the strategy most exposed to catastrophic
forgetting — see the README's discussion of Failure Mode 1.

Each model uses its native pre-training objective (not a generic surrogate),
preserving the inductive biases it was designed with:

  Chronos  — cross-entropy over quantized token IDs (language-model objective).
             Official adaptation uses scripts/training/train.py --no-random-init
             with Arrow/GluonTS format; our finetune() wraps the same CE loss
             via its HF generate() pipeline when the inner model is accessible.

  Moirai   — NLL over mixture-of-distributions patches.
             Official adaptation uses `python -m cli.train -cp conf/finetune`
             with Lightning + Uni2TS; our finetune() wraps the same NLL objective
             when the inner PatchedTransformer is accessible.

  TimesFM  — next-patch MSE with RevIN instance normalization (Das et al. 2024).
             RevIN is applied per-sample in _run_epoch_1x / _run_epoch_2p5:
             x_norm = (x - μ) / (σ + ε); output denormalized before loss.
             Causal autoregressive; all valid context prefix lengths sampled.

  PatchTST — two-phase domain adaptation (Nie et al. ICLR 2023):
               Phase 1 — masked patch reconstruction (40% patches zeroed BERT-
               style); MSE loss only on masked positions; encoder + recon_head
               trained. Uses PatchMaskedWindowDataset.
               Phase 2 — forecast fine-tuning; encoder weights retained from
               Phase 1; forecast head trained on standard next-H-step MSE.

  TTM/Granite — patch-level MSE over multi-scale patch embeddings.

MaskedWindowDataset is used as context-level data augmentation for Chronos,
Moirai, and TimesFM (temporal + span + feature masking on input context).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.loaders import (
    PatchMaskedWindowDataset,
    build_window_datasets,
    get_active_link_indices,
    load_all_splits,
    load_datasets,
)
from evaluation.plots import plot_experiment3_finetune, plot_training_history
from experiments.exp2_zeroshot import evaluate_model_zeroshot
from models import MODELS


def run(
    cfg: dict,
    out_dir: Path,
    model_names: list[str] | None = None,
    zeroshot_results: dict | None = None,
) -> dict:
    print("\n=== Experiment 3 (Full DA): Stage 1 Domain Adaptation ===")
    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]

    splits   = load_all_splits(telemetry, datasets.get("split_info"))
    da_tel   = splits["domain_adapt"]   # Stage 1 self-supervised training
    test_tel = splits["test"]           # Stage 1 post-adaptation task evaluation

    # Split domain_adapt internally: 85% training, 15% internal early-stopping val.
    # The global val/test splits are reserved for Experiment 4 and are never seen here.
    # Need at least context_len + pred_len steps in val to generate windows;
    # 10% of domain_adapt (561 steps) falls short of the 576 minimum, so use 15%.
    n_da         = len(da_tel)
    n_da_val     = max(int(n_da * 0.15), cfg["model"]["context_len"] + cfg["model"]["prediction_len"] + 50)
    da_train_tel = da_tel.iloc[: n_da - n_da_val]
    da_val_tel   = da_tel.iloc[n_da - n_da_val :]

    # Active links for evaluation (mean util ≥ 0.5%) — full telemetry for consistency
    active_idx = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    # DA training uses a lower threshold to include more links for richer signal.
    da_thresh        = cfg["data"].get("da_link_threshold", cfg["data"].get("active_link_threshold", 0.005))
    da_idx           = get_active_link_indices(da_train_tel, da_thresh)
    active_cols      = da_train_tel.columns[da_idx]
    da_train_tel_act = da_train_tel[active_cols]
    da_val_tel_act   = da_val_tel[active_cols]

    # Stride strategy for DA:
    # Use stride_da (default 4) for all models to get ~3x more windows from
    # active links compared to stride=10. Dense overlap is acceptable for
    # self-supervised DA since each window receives independent random masks.
    ft_stride_da = cfg["finetune"].get("stride_da", 4)
    ft_stride_pt = cfg["finetune"].get("stride_da", 4)

    train_ds_fm, val_ds_fm, _, _ = build_window_datasets(
        da_train_tel_act, da_val_tel_act, da_val_tel_act,
        context_len=cfg["model"]["context_len"],
        prediction_len=cfg["model"]["prediction_len"],
        capacity_gbps=cfg["data"]["capacity_gbps"],
        masked=True,
        stride=ft_stride_da,
    )
    train_ds_pt, val_ds_pt, _, _ = build_window_datasets(
        da_train_tel_act, da_val_tel_act, da_val_tel_act,
        context_len=cfg["model"]["context_len"],
        prediction_len=cfg["model"]["prediction_len"],
        capacity_gbps=cfg["data"]["capacity_gbps"],
        masked=True,
        stride=ft_stride_pt,
    )
    print(f"  Domain adapt windows — foundation models: {len(train_ds_fm)} "
          f"(stride={ft_stride_da}, {len(active_cols)} active links)")
    print(f"  Domain adapt windows — PatchTST:          {len(train_ds_pt)} "
          f"(stride={ft_stride_pt}, {len(active_cols)} active links)")

    all_results   = {}
    all_histories = {}
    names       = model_names or ["chronos", "timesfm", "moirai", "patchtst", "ttm"]
    weights_dir = Path(cfg["paths"]["weights"])

    _ft_batch = cfg["finetune"]["batch_size"]
    _fm_batch = max(8, _ft_batch // 4)

    for model_name in names:
        print(f"\n  Fine-tuning {model_name} …")
        gc.collect()
        torch.cuda.empty_cache()
        model_cls = MODELS[model_name]

        if model_name == "patchtst":
            extra_kw  = {k: cfg["model"][k] for k in ("patch_len", "d_model", "n_heads", "n_layers", "dropout")}
            batch_size = _ft_batch
        elif model_name in ("chronos", "moirai"):
            extra_kw   = {"model_size": cfg.get("model_variant", {}).get(model_name, "large")}
            batch_size = _fm_batch
        else:
            extra_kw   = {}
            batch_size = _fm_batch

        # Dataset selection: PatchTST uses denser stride; foundation models use 24h windows.
        if model_name == "patchtst":
            _train_ds, _val_ds = train_ds_pt, val_ds_pt
        else:
            _train_ds, _val_ds = train_ds_fm, val_ds_fm
        train_loader = DataLoader(_train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(_val_ds,   batch_size=batch_size)

        model = model_cls(
            context_len=cfg["model"]["context_len"],
            prediction_len=cfg["model"]["prediction_len"],
            device=None,
            **extra_kw,
        )
        model.load()

        # Foundation models: 0.3× base LR — more adaptation signal than 0.1×,
        # still conservative enough to avoid catastrophic forgetting.
        base_lr = cfg["finetune"]["learning_rate"]
        if model_name == "patchtst":
            ft_lr = base_lr
        else:
            ft_lr = base_lr * 0.3

        if model_name == "patchtst":
            # Phase 1 — masked patch reconstruction (Nie et al. ICLR 2023).
            # PatchMaskedWindowDataset: (masked_ctx, original_ctx, patch_mask)
            # 40% of patches zeroed BERT-style; MSE loss only on masked positions.
            patch_len = cfg["model"].get("patch_len", 16)
            mask_ratio = cfg["finetune"].get("patch_mask_ratio", 0.40)
            pt_pretrain_ds = PatchMaskedWindowDataset(
                _train_ds.base if hasattr(_train_ds, "base") else _train_ds,
                patch_len=patch_len,
                mask_ratio=mask_ratio,
            )
            pt_pretrain_val_ds = PatchMaskedWindowDataset(
                _val_ds.base if hasattr(_val_ds, "base") else _val_ds,
                patch_len=patch_len,
                mask_ratio=mask_ratio,
            )
            pretrain_loader = DataLoader(pt_pretrain_ds, batch_size=batch_size, shuffle=True)
            pretrain_val_loader = DataLoader(pt_pretrain_val_ds, batch_size=batch_size)

            print(f"  [{model_name}] Phase 1: masked patch pre-training …")
            pre_hist = model.pretrain(
                train_loader=pretrain_loader,
                val_loader=pretrain_val_loader,
                epochs=cfg["finetune"].get("pretrain_epochs", cfg["finetune"]["epochs"]),
                lr=base_lr,
                weight_decay=cfg["finetune"]["weight_decay"],
                patience=cfg["finetune"]["patience"],
                weights_dir=weights_dir,
            )

            print(f"  [{model_name}] Phase 2: forecast fine-tuning …")
            # Phase 2 uses the standard forecast loader (not PatchMaskedWindowDataset)
            ft_hist = model.finetune(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg["finetune"]["epochs"],
                lr=ft_lr * 0.1,   # lower LR — encoder already warm from Phase 1
                weight_decay=cfg["finetune"]["weight_decay"],
                patience=cfg["finetune"]["patience"],
                weights_dir=weights_dir,
            )
            history = {
                "pretrain_loss":     pre_hist["pretrain_loss"],
                "pretrain_val_loss": pre_hist["pretrain_val_loss"],
                "train_loss":        ft_hist["train_loss"],
                "val_loss":          ft_hist["val_loss"],
            }
        else:
            history = model.finetune(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg["finetune"]["epochs"],
                lr=ft_lr,
                weight_decay=cfg["finetune"]["weight_decay"],
                patience=cfg["finetune"]["patience"],
                weights_dir=weights_dir,
            )

        all_histories[model_name] = history

        weights_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(weights_dir / f"{model_name}_fulldA.pt")

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
                print(f"    {model_name}_fulldA / {task}: MASE={mase:.3f}  Skill_SN={sk:.3f}")
            elif task == "t2_exceedance":
                ci = m.get("t2_auroc_ci", [float("nan"), float("nan")])
                print(f"    {model_name}_fulldA / {task}: "
                      f"AUROC={m.get('t2_auroc', float('nan')):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
                      f"BSS={m.get('t2_bss', float('nan')):.3f}")
            elif task == "t4_asym":
                mae  = m.get("t4_mae", float("nan"))
                sk_p = m.get("t4_skill_persist", float("nan"))
                sk_s = m.get("t4_skill_sn", float("nan"))
                print(f"    {model_name}_fulldA / {task}: MAE={mae:.4f}  "
                      f"Skill_P={sk_p:.3f}  Skill_SN={sk_s:.3f}")
            else:  # t3_timing
                mae = m.get("mae_h", float("nan"))
                sk  = m.get("t3_skill", float("nan"))
                skp = m.get("t3_skill_persist", float("nan"))
                ci  = m.get("mae_ci", [float("nan"), float("nan")])
                print(f"    {model_name}_fulldA / {task}: Skill(rand)={sk:.3f}  "
                      f"Skill(persist)={skp:.3f}  MAE={mae:.2f}h [{ci[0]:.2f}, {ci[1]:.2f}]")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, new_data in [("exp3_fulldA_results.json", all_results),
                             ("exp3_fulldA_histories.json", all_histories)]:
        cache = out_dir / fname
        merged = json.loads(cache.read_text()) if cache.exists() else {}
        merged.update(new_data)
        cache.write_text(json.dumps(merged, indent=2))

    plot_training_history(all_histories, out_dir=str(out_dir / "figures"))

    if zeroshot_results is not None:
        def _metric(task, d):
            if task == "t1_trajectory":
                return d.get("t1_skill_sn", 0.0)
            elif task == "t2_exceedance":
                return d.get("t2_auroc", 0.0)
            elif task == "t4_asym":
                return d.get("t4_skill_persist", 0.0)
            else:
                return d.get("t3_skill", 0.0)

        zs_scores = {m: {t: _metric(t, v) for t, v in tasks.items()}
                     for m, tasks in zeroshot_results.items()}
        ft_scores = {m: {t: _metric(t, v) for t, v in tasks.items()}
                     for m, tasks in all_results.items()}
        plot_experiment3_finetune(zs_scores, ft_scores, out_dir=str(out_dir / "figures"))

    print(f"  Results saved to {out_dir}/exp3_fulldA_results.json")
    return all_results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
