"""Experiment 5 (RevIN ablation) — Reversible Instance Normalization, zero-shot.

Diagnostic ablation (paper §6.2) that rules out scale/distribution shift as
the explanation for the Full DA / LP-FT exceedance gap. RevIN (Kim et al.
2022) normalizes each context window to zero mean / unit std before model
inference, then denormalizes the output.

Hypothesis: if RevIN helps, the pretrained TSFM representations are
scale-agnostic and the main gap is distributional (magnitude mismatch). If
not, the gap is elsewhere (e.g. temporal structure, or the label-task
mismatch identified in §5.3).

Reported result: applying RevIN to all five TSFMs yields T2 AUROC mean =
0.229 vs. baseline ZS mean = 0.224 (delta = +0.005, noise level) — scale
normalization does not explain the failure.

Note: TimesFM already applies RevIN internally during pretraining (Das et
al. 2024). Wrapping TimesFM here adds double normalization — its result
should be interpreted accordingly.

Outputs: exp5_revin_results.json
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import yaml

from data.loaders import get_active_link_indices, load_all_splits, load_datasets
from experiments.exp2_zeroshot import evaluate_model_zeroshot
from models import MODELS
from models.base import TSFMBase


class RevINWrapper(TSFMBase):
    """Applies per-channel, per-window instance normalization around any TSFM.

    forecast(ctx) normalizes ctx to (μ=0, σ=1) per channel, calls the inner
    model, then denormalizes the output with the same stats.

    forecast_samples_many applies the same transform to each context before
    sampling, then denormalizes each sample array.
    """

    name: str = ""  # inherits inner model name

    def __init__(self, model: TSFMBase) -> None:
        # Bypass TSFMBase.__init__ — we delegate everything to inner
        self._inner = model
        self.name = model.name
        self.context_len = model.context_len
        self.prediction_len = model.prediction_len
        self.device = model.device
        self._loaded = True

    # TSFMBase abstract methods
    def load(self) -> None:
        pass  # already loaded via _inner

    def forecast(self, context: np.ndarray) -> np.ndarray:
        ctx = np.asarray(context, dtype=np.float32)
        mean = ctx.mean(axis=0, keepdims=True)           # (1,) or (1, C)
        std  = np.maximum(ctx.std(axis=0, keepdims=True), 1e-8)
        pred_norm = np.asarray(self._inner.forecast((ctx - mean) / std))
        return pred_norm * std[-1] + mean[-1]            # denorm with same stats

    def forecast_many(self, contexts: list[np.ndarray]) -> list[np.ndarray]:
        return [self.forecast(c) for c in contexts]

    def forecast_samples_many(
        self,
        contexts: list[np.ndarray],
        **kwargs,
    ) -> list[np.ndarray]:
        results = []
        for ctx in contexts:
            ctx = np.asarray(ctx, dtype=np.float32)
            mean = ctx.mean(axis=0, keepdims=True)
            std  = np.maximum(ctx.std(axis=0, keepdims=True), 1e-8)
            samples = self._inner.forecast_samples_many(
                [(ctx - mean) / std], **kwargs
            )[0]                                         # (n_samples, pred_len)
            results.append(np.asarray(samples) * std.ravel()[-1] + mean.ravel()[-1])
        return results


def run(cfg: dict, out_dir: Path, model_names: list[str] | None = None) -> dict:
    print("\n=== Experiment 5 (RevIN ablation): Zero-shot Evaluation ===")
    datasets  = load_datasets(cfg["data"])
    telemetry = datasets["telemetry"]
    splits    = load_all_splits(telemetry, datasets.get("split_info"))
    test_tel  = splits["test"]

    active_idx = get_active_link_indices(
        telemetry, cfg["data"].get("active_link_threshold", 0.005)
    )

    names = model_names or ["chronos", "timesfm", "moirai", "patchtst", "ttm"]
    all_results: dict = {}

    for model_name in names:
        print(f"\n  [{model_name}] loading …")
        gc.collect()
        model_cls = MODELS[model_name]
        extra_kw: dict = {}
        if model_name in ("chronos", "moirai"):
            extra_kw["model_size"] = cfg.get("model_variant", {}).get(model_name, "large")
        elif model_name == "patchtst":
            extra_kw = {k: cfg["model"][k] for k in ("patch_len", "d_model", "n_heads", "n_layers", "dropout")}
        elif model_name == "ttm":
            extra_kw["model_size"] = cfg.get("model_variant", {}).get("ttm", "r2")

        model = model_cls(
            context_len=cfg["model"]["context_len"],
            prediction_len=cfg["model"]["prediction_len"],
            device=None,
            **extra_kw,
        )
        model.load()

        if model_name == "timesfm":
            print(f"    NOTE: TimesFM already uses RevIN internally — this applies double normalization")

        wrapped = RevINWrapper(model)

        model_results = evaluate_model_zeroshot(
            wrapped, cfg, test_tel, active_idx,
            eval_start=cfg["model"].get("eval_horizon_start", 0),
            model_name=model_name,
        )
        all_results[model_name] = model_results

        del model, wrapped
        gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    cache  = out_dir / "exp5_revin_results.json"
    merged = json.loads(cache.read_text()) if cache.exists() else {}
    merged.update(all_results)
    cache.write_text(json.dumps(merged, indent=2))
    print(f"\n  Results saved → {cache}")
    return merged


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    run(cfg, Path(cfg["paths"]["results"]))
