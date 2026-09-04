"""Experiment 3 (ELoRA) — Extended LoRA domain adaptation.

Same pipeline as exp3_lora.py (LoRA) but with a wider set of adapted modules
(paper §3.2, "LoRA extended to FFN layers, ≈1.2% trainable"):
  - Attention: Q, K, V, O projections (same as exp3_lora.py)
  - FFN: fc1 / fc2 / wi / wo / gate_proj / up_proj / down_proj
  - Output projection: output_patch_embedding.output_layer (critical for Chronos/Moirai
    to re-calibrate the patch decoder for the AmLight utilization distribution)

Also changes alpha = r (1:1 ratio instead of 2:1) following the Chronos-2
fine-tuning guide, which found ratio 1:1 more stable for out-of-domain data.

Rationale for the wider adapter set:
  - Attention-only LoRA leaves the patch decoder misaligned with
    AmLight's 0-100 Gbps scale
  - output_patch_embedding.output_layer is the bottleneck for distribution shift
  - FFN layers encode domain-specific feature interactions; adapting them
    complements attention-level changes (Chronos-2 guide, arXiv 2405.10216)

Outputs:
  - {model}_elora.pt  (extended LoRA adapter weights)
  - exp3_elora_results.json
  - exp3_elora_histories.json
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
from models.lora_utils import (
    LoRALinear,
    get_inner_module,
    lora_state_dict,
)
from models.moirai import _run_epoch as _moirai_epoch
from models.timesfm import _run_epoch_1x as _timesfm_epoch


_LORA_R     = 8
_LORA_ALPHA = 8.0   # ratio 1:1 (vs 2:1 in exp3_lora.py) — more stable for OOD

_LORA_MODELS = {"chronos", "moirai", "timesfm"}

# Extended target module names: attention + FFN + output projections
_EXTENDED_NAMES: frozenset[str] = frozenset({
    # --- Attention ---
    "q", "k", "v", "o",
    "q_proj", "k_proj", "v_proj", "out_proj",
    "query", "key", "value",
    "query_proj", "key_proj", "value_proj",
    "Wq", "Wk", "Wv", "Wo",
    "in_proj", "c_attn",
    "linear_q", "linear_k", "linear_v",
    "qkv", "proj",
    # --- FFN ---
    "fc1", "fc2",          # standard BERT/ViT FFN
    "wi", "wo",            # T5 dense FFN
    "wi_0", "wi_1",        # T5 gated FFN (SwiGLU)
    "c_fc",                # GPT-2 FFN
    "gate_proj", "up_proj", "down_proj",  # LLaMA/Mistral
    "dense", "dense_h_to_4h", "dense_4h_to_h",  # BERT
    # --- Output / patch projection ---
    "output_layer",        # Chronos output_patch_embedding.output_layer
    "output_proj",
    "head",
})

_EXTENDED_CLASS_KW: tuple[str, ...] = ("attention", "attn", "mlp", "ffn", "feed_forward")


def inject_lora_extended(module: nn.Module, r: int = 8, alpha: float = 8.0) -> int:
    """Like inject_lora but targets FFN and output layers in addition to attention."""
    injected   = 0
    parent_cls = type(module).__name__.lower()
    parent_hit = any(kw in parent_cls for kw in _EXTENDED_CLASS_KW)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (name in _EXTENDED_NAMES or parent_hit):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            injected += 1
        else:
            injected += inject_lora_extended(child, r=r, alpha=alpha)
    return injected


# ── Training helpers (identical to exp3_lora.py except they use inject_lora_extended) ─

def _generic_train_loop(model_name, inner_model, epochs, patience, weights_dir,
                        optimizer, scheduler, train_fn, val_fn):
    best_pt  = Path(weights_dir) / f"{model_name}_elora_best_tmp.pt"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf");  no_improve = 0

    for epoch in range(epochs):
        inner_model.train() if isinstance(inner_model, nn.Module) else None
        train_loss = train_fn(optimizer)
        inner_model.eval() if isinstance(inner_model, nn.Module) else None
        val_loss = val_fn()
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  [{model_name}] elora ep {epoch+1:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss;  no_improve = 0
            torch.save(lora_state_dict(inner_model), best_pt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{model_name}] elora early stop at epoch {epoch+1}")
                break

    if best_pt.exists():
        sd = torch.load(best_pt, map_location="cpu")
        cur = dict(inner_model.state_dict())
        cur.update(sd)
        inner_model.load_state_dict(cur)
        best_pt.unlink()
    return history


def run(cfg: dict, out_dir: Path, model_names: list[str] | None = None) -> dict:
    print("\n=== Experiment 3 (ELoRA): Extended LoRA Domain Adaptation ===")
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

    active_idx = get_active_link_indices(
        telemetry, cfg["data"].get("active_link_threshold", 0.005)
    )
    da_thresh   = cfg["data"].get("da_link_threshold",
                                  cfg["data"].get("active_link_threshold", 0.005))
    da_idx      = get_active_link_indices(da_train_tel, da_thresh)
    active_cols = da_train_tel.columns[da_idx]
    da_train_act = da_train_tel[active_cols]
    da_val_act   = da_val_tel[active_cols]

    ft_stride_da = cfg["finetune"].get("stride_da", 4)
    train_ds, val_ds, _, _ = build_window_datasets(
        da_train_act, da_val_act, da_val_act,
        context_len=cfg["model"]["context_len"],
        prediction_len=cfg["model"]["prediction_len"],
        capacity_gbps=cfg["data"]["capacity_gbps"],
        masked=True, stride=ft_stride_da,
    )
    print(f"  ELoRA windows: {len(train_ds)} train / {len(val_ds)} val "
          f"(stride={ft_stride_da}, {len(active_cols)} links, "
          f"r={_LORA_R} alpha={_LORA_ALPHA})")

    weights_dir = Path(cfg["paths"]["weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    base_lr   = cfg["finetune"]["learning_rate"]
    _fm_batch = max(8, cfg["finetune"]["batch_size"] // 4)

    names        = model_names or ["chronos", "timesfm", "moirai"]
    all_results: dict   = {}
    all_histories: dict = {}

    for model_name in names:
        if model_name not in _LORA_MODELS:
            print(f"\n  [{model_name}] skipped — ELoRA not applicable")
            continue

        print(f"\n  ELoRA fine-tuning {model_name} …")
        gc.collect();  torch.cuda.empty_cache()

        model_cls = MODELS[model_name]
        extra_kw  = {}
        if model_name in ("chronos", "moirai"):
            extra_kw = {"model_size": cfg.get("model_variant", {}).get(model_name, "large")}

        model = model_cls(
            context_len=cfg["model"]["context_len"],
            prediction_len=cfg["model"]["prediction_len"],
            device=None, **extra_kw,
        )
        model.load()

        inner = get_inner_module(model_name, model)
        for p in inner.parameters():
            p.requires_grad = False
        n_lora = inject_lora_extended(inner, r=_LORA_R, alpha=_LORA_ALPHA)
        if n_lora == 0:
            print(f"  [{model_name}] WARNING: 0 ELoRA layers — re-enabling all params")
            for p in inner.parameters():
                p.requires_grad = True
        else:
            trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in inner.parameters())
            print(f"  [{model_name}] ELoRA: {n_lora} layers, "
                  f"{trainable:,}/{total:,} params ({100*trainable/max(total,1):.2f}%)")

        train_loader = DataLoader(train_ds, batch_size=_fm_batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=_fm_batch)

        tokenizer = getattr(getattr(model, "_pipeline", None), "tokenizer", None)
        criterion = nn.MSELoss()

        lora_params = [p for p in inner.parameters() if p.requires_grad]
        optimizer   = torch.optim.AdamW(lora_params, lr=base_lr,
                                        weight_decay=cfg["finetune"]["weight_decay"])
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["finetune"]["epochs"]
        )

        if model_name == "chronos" and tokenizer is not None:
            history = _generic_train_loop(
                "chronos", inner, cfg["finetune"]["epochs"],
                cfg["finetune"]["patience"], weights_dir, optimizer, scheduler,
                train_fn=lambda opt: _epoch_loss_chronos(inner, tokenizer, train_loader, opt, model.device),
                val_fn=lambda: _epoch_loss_chronos(inner, tokenizer, val_loader, None, model.device),
            )
        elif model_name == "moirai":
            moirai_forecast = model._model
            history = _generic_train_loop(
                "moirai", moirai_forecast, cfg["finetune"]["epochs"],
                cfg["finetune"]["patience"], weights_dir, optimizer, scheduler,
                train_fn=lambda opt: _moirai_epoch(moirai_forecast, train_loader, opt, criterion, model.device),
                val_fn=lambda: _moirai_epoch(moirai_forecast, val_loader, None, criterion, model.device),
            )
        elif model_name == "timesfm":
            if getattr(model, "_api_version", "1.x") != "1.x":
                print(f"  [timesfm] ELoRA only for 1.x API; skipping")
                del model; continue
            torch_model = model._get_torch_module().to(model.device)
            history = _generic_train_loop(
                "timesfm", torch_model, cfg["finetune"]["epochs"],
                cfg["finetune"]["patience"], weights_dir, optimizer, scheduler,
                train_fn=lambda opt: _timesfm_epoch(torch_model, train_loader, opt, criterion, model.device),
                val_fn=lambda: _timesfm_epoch(torch_model, val_loader, None, criterion, model.device),
            )

        all_histories[model_name] = history

        lora_path = weights_dir / f"{model_name}_elora.pt"
        torch.save(lora_state_dict(inner), lora_path)
        print(f"  [{model_name}] ELoRA weights → {lora_path} "
              f"({lora_path.stat().st_size // 1024} KB)")

        model_results = evaluate_model_zeroshot(
            model, cfg, test_tel, active_idx,
            eval_start=cfg["model"].get("eval_horizon_start", 0),
            model_name=model_name,
        )
        all_results[model_name] = model_results

        for task, m in model_results.items():
            if task == "t1_trajectory":
                print(f"    {model_name}_elora / {task}: "
                      f"MASE={m.get('t1_mase', float('nan')):.3f}  "
                      f"Skill_SN={m.get('t1_skill_sn', float('nan')):.3f}")
            elif task == "t2_exceedance":
                ci = m.get("t2_auroc_ci", [float("nan"), float("nan")])
                print(f"    {model_name}_elora / {task}: "
                      f"AUROC={m.get('t2_auroc', float('nan')):.3f} "
                      f"[{ci[0]:.3f}, {ci[1]:.3f}]  "
                      f"BSS={m.get('t2_bss', float('nan')):.3f}")
            elif task == "t4_asym":
                print(f"    {model_name}_elora / {task}: "
                      f"MAE={m.get('t4_mae', float('nan')):.4f}  "
                      f"Skill_P={m.get('t4_skill_persist', float('nan')):.3f}")
            else:  # t3_timing
                mae = m.get("mae_h", float("nan"))
                sk  = m.get("t3_skill", float("nan"))
                skp = m.get("t3_skill_persist", float("nan"))
                print(f"    {model_name}_elora / {task}: "
                      f"Skill(rand)={sk:.3f}  Skill(persist)={skp:.3f}  MAE={mae:.2f}h")

        del model;  gc.collect();  torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in [("exp3_elora_results.json", all_results),
                        ("exp3_elora_histories.json", all_histories)]:
        cache  = out_dir / fname
        merged = json.loads(cache.read_text()) if cache.exists() else {}
        merged.update(data)
        cache.write_text(json.dumps(merged, indent=2))

    print(f"  Results saved to {out_dir}/exp3_elora_results.json")
    return all_results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
