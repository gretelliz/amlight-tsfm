"""Experiment 6 (long-context ablation) — zero-shot, context_len=672h (4 weeks).

Diagnostic ablation (paper §6.2) that rules out a too-short context window
as the explanation for the Full DA / LP-FT exceedance gap. Tests whether
TSFMs benefit from a longer lookback window that captures full weekly
periodicity of AmLight traffic. At the default context_len=336h (2 weeks),
models see ~2 diurnal/weekly cycles; at 672h (4 weeks), they see ~4 cycles.

Hypothesis: AmLight has science-driven weekly periodicities (LHC schedules,
telescope campaigns, university academic calendar) that require 4+ weeks of
history to reliably characterize. A 2-week context may be too short.

Reported result: the long-context model provides 15x less improvement on
T2/T3 than five auto-labeled examples (ZS+MLP@5, exp4_fewshot.py) — context
length is not the bottleneck; labeled task supervision is categorically
more informative than additional unlabeled history.

TTM-R2 has a native context of 512h — it pads shorter inputs and crops
longer ones to 512h internally, so its results here may not differ from the
Experiment 2 baseline. Other models (Chronos, TimesFM, Moirai, PatchTST)
support flexible context lengths.

Outputs: exp6_longcontext_results.json
"""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import yaml

from data.loaders import get_active_link_indices, load_all_splits, load_datasets
from experiments.exp2_zeroshot import evaluate_model_zeroshot
from models import MODELS

_LONG_CTX = 672   # 4 weeks at hourly resolution


def run(cfg: dict, out_dir: Path, model_names: list[str] | None = None) -> dict:
    print(f"\n=== Experiment 6 (long-context ablation): Zero-shot (ctx={_LONG_CTX}h) ===")

    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]
    splits    = load_all_splits(telemetry, datasets.get("split_info"))
    test_tel  = splits["test"]

    active_idx = get_active_link_indices(
        telemetry, cfg["data"].get("active_link_threshold", 0.005)
    )

    # Override context_len in a local copy of cfg so evaluate_model_zeroshot
    # uses 672h for window creation while all other settings remain unchanged.
    cfg_long = copy.deepcopy(cfg)
    cfg_long["model"]["context_len"] = _LONG_CTX

    names = model_names or ["chronos", "timesfm", "moirai", "patchtst", "ttm"]
    all_results: dict = {}

    for model_name in names:
        print(f"\n  [{model_name}] loading (ctx={_LONG_CTX}h) …")
        gc.collect()

        model_cls = MODELS[model_name]
        extra_kw: dict = {}
        if model_name in ("chronos", "moirai"):
            extra_kw["model_size"] = cfg.get("model_variant", {}).get(model_name, "large")
        elif model_name == "patchtst":
            extra_kw = {k: cfg["model"][k] for k in ("patch_len", "d_model", "n_heads", "n_layers", "dropout")}
        elif model_name == "ttm":
            extra_kw["model_size"] = cfg.get("model_variant", {}).get("ttm", "r2")
            print(f"    NOTE: TTM native context=512h — crops 672h input to 512h; "
                  f"results may match exp2 baseline")

        try:
            model = model_cls(
                context_len=_LONG_CTX,
                prediction_len=cfg["model"]["prediction_len"],
                device=None,
                **extra_kw,
            )
            model.load()
        except Exception as e:
            print(f"    [{model_name}] SKIP — failed to load with ctx={_LONG_CTX}: {e}")
            continue

        model_results = evaluate_model_zeroshot(
            model, cfg_long, test_tel, active_idx,
            eval_start=cfg["model"].get("eval_horizon_start", 0),
            model_name=model_name,
        )
        all_results[model_name] = model_results

        del model
        gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    cache  = out_dir / "exp6_longcontext_results.json"
    merged = json.loads(cache.read_text()) if cache.exists() else {}
    merged.update(all_results)
    cache.write_text(json.dumps(merged, indent=2))
    print(f"\n  Results saved → {cache}")
    return merged


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
