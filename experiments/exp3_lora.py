"""Experiment 3 (LoRA) — Low-Rank Adaptation domain adaptation on AmLight telemetry.

Same training data and splits as exp3_full_da.py (full fine-tuning), but uses
Low-Rank Adaptation (LoRA, Hu et al. 2022) instead of updating all model
parameters. Base weights are frozen; only small A and B matrices are trained
in the attention projections (r=8, ~0.6% of parameters — paper §3.2).

Advantages over full fine-tuning:
  - No catastrophic forgetting — base knowledge is preserved
  - 10-100x fewer trainable parameters
  - Higher learning rate tolerable since the base cannot drift

Applied to the three pre-trained foundation models (Chronos, Moirai, TimesFM).
PatchTST is skipped: it has no pre-trained knowledge to preserve, so LoRA and
full fine-tuning are equivalent for a randomly-initialised model.

Referenced in the paper's "LoRA and ELoRA" discussion (§5.3) alongside
exp3_elora.py; T1/T2 are not re-evaluated there since the backbone-update
mechanism (and forgetting pattern) is architecture-determined, not
task-determined — only T3/T4 are discussed.

Outputs:
  - {model}_lora.pt  (LoRA adapter weights only — small files)
  - exp3_lora_results.json  (per-model task metrics, same schema as exp3_full_da)
  - exp3_lora_histories.json
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
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
from models.chronos import _epoch_loss_chronos
from models.lora_utils import get_inner_module, inject_lora, lora_state_dict
from models.moirai import _run_epoch as _moirai_epoch
from models.timesfm import _run_epoch_1x as _timesfm_epoch


_LORA_R     = 8
_LORA_ALPHA = 16.0

# LoRA applies only to pre-trained foundation models.
_LORA_MODELS = {"chronos", "moirai", "timesfm"}


def _train_lora_chronos(model, inner_model, train_loader, val_loader, epochs, lr,
                        weight_decay, patience, weights_dir, device):
    tokenizer = getattr(model._pipeline, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "context_input_transform"):
        print(f"  [chronos] LoRA skipped — tokenizer API unavailable")
        return {"train_loss": [], "val_loss": []}

    lora_params = [p for p in inner_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return _generic_train_loop(
        model_name="chronos", inner_model=inner_model, epochs=epochs,
        patience=patience, weights_dir=weights_dir, optimizer=optimizer,
        scheduler=scheduler,
        train_fn=lambda opt: _epoch_loss_chronos(inner_model, tokenizer, train_loader, opt, device),
        val_fn=lambda: _epoch_loss_chronos(inner_model, tokenizer, val_loader, None, device),
    )


def _train_lora_moirai(model, inner_module, train_loader, val_loader, epochs, lr,
                       weight_decay, patience, weights_dir, device):
    criterion = nn.MSELoss()
    lora_params = [p for p in inner_module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # _moirai_epoch expects MoiraiForecast (model._model), uses .module internally
    moirai_forecast = model._model
    return _generic_train_loop(
        model_name="moirai", inner_model=moirai_forecast, epochs=epochs,
        patience=patience, weights_dir=weights_dir, optimizer=optimizer,
        scheduler=scheduler,
        train_fn=lambda opt: _moirai_epoch(moirai_forecast, train_loader, opt, criterion, device),
        val_fn=lambda: _moirai_epoch(moirai_forecast, val_loader, None, criterion, device),
    )


def _train_lora_timesfm(model, torch_model, train_loader, val_loader, epochs, lr,
                        weight_decay, patience, weights_dir, device):
    if getattr(model, "_api_version", "1.x") != "1.x":
        print(f"  [timesfm] LoRA only supported with API 1.x; skipping")
        return {"train_loss": [], "val_loss": []}

    criterion = nn.MSELoss()
    lora_params = [p for p in torch_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return _generic_train_loop(
        model_name="timesfm", inner_model=torch_model, epochs=epochs,
        patience=patience, weights_dir=weights_dir, optimizer=optimizer,
        scheduler=scheduler,
        train_fn=lambda opt: _timesfm_epoch(torch_model, train_loader, opt, criterion, device),
        val_fn=lambda: _timesfm_epoch(torch_model, val_loader, None, criterion, device),
    )


def _generic_train_loop(model_name, inner_model, epochs, patience, weights_dir,
                        optimizer, scheduler, train_fn, val_fn):
    best_pt = Path(weights_dir) / f"{model_name}_lora_best_tmp.pt"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        inner_model.train() if isinstance(inner_model, nn.Module) else None
        train_loss = train_fn(optimizer)
        inner_model.eval() if isinstance(inner_model, nn.Module) else None
        val_loss = val_fn()
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  [{model_name}] lora epoch {epoch+1:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(lora_state_dict(inner_model), best_pt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{model_name}] lora early stopping at epoch {epoch+1}")
                break

    if best_pt.exists():
        sd = torch.load(best_pt, map_location="cpu")
        cur = dict(inner_model.state_dict())
        cur.update(sd)
        inner_model.load_state_dict(cur)
        best_pt.unlink()
    return history


def run(
    cfg: dict,
    out_dir: Path,
    model_names: list[str] | None = None,
    zeroshot_results: dict | None = None,
) -> dict:
    print("\n=== Experiment 3 (LoRA): LoRA Domain Adaptation ===")
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

    # Active links for evaluation (mean util ≥ 0.5%) — full telemetry for consistency
    active_idx  = get_active_link_indices(telemetry, cfg["data"].get("active_link_threshold", 0.005))

    # LoRA training uses a lower threshold to include more links (better adaptation signal).
    da_thresh        = cfg["data"].get("da_link_threshold", cfg["data"].get("active_link_threshold", 0.05))
    da_idx           = get_active_link_indices(da_train_tel, da_thresh)
    active_cols      = da_train_tel.columns[da_idx]
    da_train_tel_act = da_train_tel[active_cols]
    da_val_tel_act   = da_val_tel[active_cols]

    ft_stride_da = cfg["finetune"].get("stride_da", 4)

    train_ds, val_ds, _, _ = build_window_datasets(
        da_train_tel_act, da_val_tel_act, da_val_tel_act,
        context_len=cfg["model"]["context_len"],
        prediction_len=cfg["model"]["prediction_len"],
        capacity_gbps=cfg["data"]["capacity_gbps"],
        masked=True,
        stride=ft_stride_da,
    )
    print(f"  LoRA adapt windows: {len(train_ds)} train / {len(val_ds)} val "
          f"(stride={ft_stride_da}, {len(active_cols)} active links)")
    weights_dir = Path(cfg["paths"]["weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    base_lr     = cfg["finetune"]["learning_rate"]
    _fm_batch   = max(8, cfg["finetune"]["batch_size"] // 4)

    names = model_names or ["chronos", "timesfm", "moirai", "patchtst"]
    all_results: dict   = {}
    all_histories: dict = {}

    for model_name in names:
        if model_name not in _LORA_MODELS:
            print(f"\n  [{model_name}] skipped — LoRA not applicable to scratch-trained models")
            continue

        print(f"\n  LoRA fine-tuning {model_name} …")
        gc.collect()
        torch.cuda.empty_cache()

        model_cls = MODELS[model_name]
        extra_kw  = {}
        if model_name in ("chronos", "moirai"):
            extra_kw = {"model_size": cfg.get("model_variant", {}).get(model_name, "large")}

        model = model_cls(
            context_len=cfg["model"]["context_len"],
            prediction_len=cfg["model"]["prediction_len"],
            device=None,
            **extra_kw,
        )
        model.load()

        inner = get_inner_module(model_name, model)
        # Freeze all base parameters before injecting LoRA adapters.
        # Without this, non-attention params (FFN, norms, embeddings) remain
        # trainable and LoRA loses its parameter-efficiency advantage.
        for p in inner.parameters():
            p.requires_grad = False
        n_lora = inject_lora(inner, r=_LORA_R, alpha=_LORA_ALPHA)
        if n_lora == 0:
            print(f"  [{model_name}] WARNING: 0 LoRA layers injected — "
                  f"check attention layer names")
            # Re-enable all params so training is not a no-op
            for p in inner.parameters():
                p.requires_grad = True
        else:
            trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in inner.parameters())
            print(f"  [{model_name}] LoRA: {n_lora} layers, "
                  f"{trainable:,}/{total:,} params trainable "
                  f"({100*trainable/max(total,1):.2f}%)")

        train_loader = DataLoader(train_ds, batch_size=_fm_batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=_fm_batch)

        # LoRA uses full base_lr — base weights are frozen so no catastrophic forgetting.
        if model_name == "chronos":
            history = _train_lora_chronos(
                model, inner, train_loader, val_loader,
                epochs=cfg["finetune"]["epochs"], lr=base_lr,
                weight_decay=cfg["finetune"]["weight_decay"],
                patience=cfg["finetune"]["patience"],
                weights_dir=weights_dir, device=model.device,
            )
        elif model_name == "moirai":
            history = _train_lora_moirai(
                model, inner, train_loader, val_loader,
                epochs=cfg["finetune"]["epochs"], lr=base_lr,
                weight_decay=cfg["finetune"]["weight_decay"],
                patience=cfg["finetune"]["patience"],
                weights_dir=weights_dir, device=model.device,
            )
        elif model_name == "timesfm":
            torch_model = model._get_torch_module().to(model.device)
            history = _train_lora_timesfm(
                model, torch_model, train_loader, val_loader,
                epochs=cfg["finetune"]["epochs"], lr=base_lr,
                weight_decay=cfg["finetune"]["weight_decay"],
                patience=cfg["finetune"]["patience"],
                weights_dir=weights_dir, device=model.device,
            )

        all_histories[model_name] = history

        # Save only LoRA adapter weights (small file)
        lora_path = weights_dir / f"{model_name}_lora.pt"
        torch.save(lora_state_dict(inner), lora_path)
        print(f"  [{model_name}] LoRA weights saved → {lora_path} "
              f"({lora_path.stat().st_size // 1024} KB)")

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
                print(f"    {model_name}_lora / {task}: MASE={mase:.3f}  Skill_SN={sk:.3f}")
            elif task == "t3_timing":
                ci = m.get("mae_ci", [float("nan"), float("nan")])
                print(f"    {model_name}_lora / {task}: "
                      f"Skill(rand)={m.get('t3_skill', float('nan')):.3f}  "
                      f"Skill(persist)={m.get('t3_skill_persist', float('nan')):.3f}  "
                      f"MAE={m.get('mae_h', 0):.2f}h (persist={m.get('mae_persist_h', float('nan')):.2f}h) "
                      f"[{ci[0]:.2f}, {ci[1]:.2f}]")
            elif task == "t2_exceedance":
                ci = m.get("t2_auroc_ci", [float("nan"), float("nan")])
                print(f"    {model_name}_lora / {task}: "
                      f"AUROC={m.get('t2_auroc', float('nan')):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
                      f"BSS={m.get('t2_bss', float('nan')):.3f}")
            elif task == "t4_asym":
                mae  = m.get("t4_mae", float("nan"))
                sk_p = m.get("t4_skill_persist", float("nan"))
                sk_s = m.get("t4_skill_sn", float("nan"))
                print(f"    {model_name}_lora / {task}: MAE={mae:.4f}  "
                      f"Skill_P={sk_p:.3f}  Skill_SN={sk_s:.3f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, new_data in [("exp3_lora_results.json", all_results),
                             ("exp3_lora_histories.json", all_histories)]:
        cache  = out_dir / fname
        merged = json.loads(cache.read_text()) if cache.exists() else {}
        merged.update(new_data)
        cache.write_text(json.dumps(merged, indent=2))

    print(f"  Results saved to {out_dir}/exp3_lora_results.json")
    return all_results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
