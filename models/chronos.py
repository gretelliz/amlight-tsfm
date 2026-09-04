"""Chronos (Amazon) wrapper.

Chronos is a T5-based language model fine-tuned for time series forecasting.
It tokenizes real-valued time series into discrete bins and generates forecasts
autoregressively. We use the `chronos-forecasting` package from HuggingFace.

HuggingFace model: amazon/chronos-t5-large
Install: pip install chronos-forecasting
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

# Substring that uniquely identifies the prediction-length warning.
_CHRONOS_SUPPRESS_KEY = "We recommend keeping prediction length"


class _ChronosFilter(logging.Filter):
    def filter(self, record):
        try:
            return _CHRONOS_SUPPRESS_KEY not in record.getMessage()
        except Exception:
            return True


_chronos_filter = _ChronosFilter()


def _install_chronos_filter() -> None:
    """Suppress the Chronos prediction-length warning across all possible logger names.

    Chronos uses logging.getLogger(__file__) whose value is the absolute install
    path and varies across environments.  We cover every plausible binding so the
    filter works regardless of how the package is installed.
    """
    import sys

    # 1. By package/module name (most common patterns).
    for name in ("chronos", "chronos.chronos"):
        logging.getLogger(name).addFilter(_chronos_filter)

    # 2. By absolute __file__ path (what Chronos actually uses internally).
    for mod_name in ("chronos", "chronos.chronos"):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "__file__", None):
            fpath = mod.__file__
            # Normalise .pyc → .py so the name matches what the logger sees.
            if fpath.endswith(".pyc"):
                fpath = fpath[:-1]
            logging.getLogger(fpath).addFilter(_chronos_filter)

    # 3. Any logger already in the registry whose name contains "chronos".
    for name, lgr in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lgr, logging.Logger) and "chronos" in name.lower():
            lgr.addFilter(_chronos_filter)

    # 4. Root logger handlers — catches records that propagate up.
    for handler in logging.getLogger().handlers:
        handler.addFilter(_chronos_filter)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*prediction length.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*prediction length.*")

from .base import TSFMBase


class ChronosModel(TSFMBase):

    name = "chronos"
    _hf_model_id = "amazon/chronos-t5-large"

    def __init__(
        self,
        context_len: int,
        prediction_len: int,
        device: Optional[str] = None,
        model_size: str = "large",
    ) -> None:
        super().__init__(context_len, prediction_len, device)
        self.model_size = model_size
        self._hf_model_id = f"amazon/chronos-t5-{model_size}"
        self._pipeline = None

    def load(self) -> None:
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "Install Chronos: pip install chronos-forecasting\n"
                "See https://github.com/amazon-science/chronos-forecasting"
            )
        _install_chronos_filter()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._pipeline = ChronosPipeline.from_pretrained(
            self._hf_model_id,
            device_map=self.device,
            torch_dtype=dtype,
        )
        self._loaded = True

    def forecast_many(self, contexts: list[np.ndarray], batch_size: int = 16) -> list[np.ndarray]:
        """Batch multiple contexts into a single pipeline.predict() call.

        batch_size=16 keeps effective sequences (16 × 20 samples = 320) within T4 VRAM.
        """
        assert self._loaded
        results = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            ctx_tensors = [
                torch.tensor(c.flatten()[-self.context_len :], dtype=torch.float32)
                for c in batch
            ]
            ctx_batch = torch.stack(ctx_tensors)  # (B, context_len)
            with torch.no_grad():
                raw = self._pipeline.predict(
                    ctx_batch,
                    prediction_length=self.prediction_len,
                    num_samples=20,
                )
            samples = raw[0] if isinstance(raw, (tuple, list)) else raw
            medians = np.median(samples.cpu().numpy(), axis=1)  # (B, pred_len)
            results.extend(list(medians))
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return results

    def _get_samples(self, context: np.ndarray) -> np.ndarray:
        """Run Chronos inference, return samples array (n_samples, pred_len)."""
        ctx = context.flatten()[-self.context_len :]
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            result = self._pipeline.predict(
                ctx_tensor,
                prediction_length=self.prediction_len,
                num_samples=20,
            )
        samples = result[0] if isinstance(result, (tuple, list)) else result
        return samples.numpy().squeeze(0)  # (n_samples, pred_len)

    def forecast(self, context: np.ndarray) -> np.ndarray:
        assert self._loaded, "call load() first"
        return np.median(self._get_samples(context), axis=0)

    def forecast_upper(self, context: np.ndarray, q: float = 0.9) -> np.ndarray:
        """q-th quantile forecast from the predictive distribution."""
        assert self._loaded, "call load() first"
        return np.quantile(self._get_samples(context), q, axis=0)

    def forecast_samples_many(
        self, contexts: list[np.ndarray], batch_size: int = 16
    ) -> list[np.ndarray]:
        """Batched sample inference. Returns list of (n_samples, pred_len) arrays."""
        assert self._loaded
        results = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            ctx_tensors = [
                torch.tensor(c.flatten()[-self.context_len :], dtype=torch.float32)
                for c in batch
            ]
            ctx_batch = torch.stack(ctx_tensors)
            with torch.no_grad():
                raw = self._pipeline.predict(
                    ctx_batch,
                    prediction_length=self.prediction_len,
                    num_samples=20,
                )
            samples = raw[0] if isinstance(raw, (tuple, list)) else raw
            arr = samples.cpu().numpy()  # (B, n_samples, pred_len)
            results.extend([arr[k] for k in range(len(batch))])
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return results

    def _get_t5_model(self):
        """Return the underlying T5ForConditionalGeneration from the pipeline.

        ChronosPipeline.model is a ChronosModel wrapper; ChronosModel.model
        is the actual T5 that accepts the standard HF `labels` argument.
        """
        chronos_wrapper = getattr(self._pipeline, "model", None)
        if chronos_wrapper is None:
            return None
        return getattr(chronos_wrapper, "model", chronos_wrapper)

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 5,
        weights_dir: str | Path = "models/weights",
    ) -> dict[str, list[float]]:
        """Fine-tune Chronos via its tokenizer-based NLL training objective.

        Chronos uses discrete tokenization: continuous values are binned and
        the T5 model is trained on next-token prediction (NLL loss). We access
        the pipeline's tokenizer to produce input/label token IDs.
        """
        assert self._loaded, "call load() first"
        Path(weights_dir).mkdir(parents=True, exist_ok=True)

        tokenizer  = getattr(self._pipeline, "tokenizer", None)
        inner_model = self._get_t5_model()   # actual T5, accepts `labels`

        if tokenizer is None or inner_model is None or not hasattr(tokenizer, "context_input_transform"):
            print(f"  [{self.name}] fine-tuning skipped — tokenizer API not available in this version")
            return {"train_loss": [], "val_loss": []}

        optimizer = torch.optim.AdamW(inner_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            inner_model.train()
            train_loss = _epoch_loss_chronos(inner_model, tokenizer, train_loader, optimizer, self.device)
            inner_model.eval()
            val_loss = _epoch_loss_chronos(inner_model, tokenizer, val_loader, None, self.device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"  [{self.name}] epoch {epoch+1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(inner_model.state_dict(), Path(weights_dir) / f"{self.name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{self.name}] early stopping at epoch {epoch+1}")
                    break

        best_path = Path(weights_dir) / f"{self.name}_best.pt"
        if best_path.exists():
            inner_model.load_state_dict(torch.load(best_path, map_location=self.device))
        return history

    def save_weights(self, path: str | Path) -> None:
        assert self._loaded
        torch.save(self._get_t5_model().state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        assert self._loaded
        self._get_t5_model().load_state_dict(
            torch.load(path, map_location=self.device)
        )


def _epoch_loss_chronos(model, tokenizer, loader: DataLoader, optimizer, device: str,
                        sub_batch: int = 2) -> float:
    """Training epoch using Chronos tokenizer-based NLL objective.

    Chronos tokenizes continuous values into discrete bins; the T5 model is
    then trained on next-token prediction. We feed the context as input_ids
    and the target as labels so the model computes the NLL loss internally.
    One random link is sampled per window (same as Moirai/TimesFM) to avoid
    the B*n_links memory explosion in T5 self-attention (480-token sequences
    require ~28 MB/layer/sub_batch=2; 24 encoder layers × 2 = ~672 MB total).
    """
    total, count = 0.0, 0
    for context, target in loader:
        B, ctx_len, n_links = context.shape
        link_idx = torch.randint(0, n_links, (B,))
        x_all = context[torch.arange(B), :, link_idx].float()   # (B, ctx_len)
        y_all = target[torch.arange(B), :, link_idx].float()    # (B, pred_len)

        sub_losses, sub_count = 0.0, 0
        for start in range(0, B, sub_batch):
            x = x_all[start : start + sub_batch]
            y = y_all[start : start + sub_batch]

            try:
                input_ids, attention_mask, tokenizer_state = tokenizer.context_input_transform(x)
                label_ids, _ = tokenizer.label_input_transform(y, tokenizer_state)
            except Exception:
                full = torch.cat([x, y], dim=1)
                input_ids, attention_mask, _ = tokenizer.context_input_transform(full)
                label_ids = input_ids[:, x.shape[1]:]
                input_ids = input_ids[:, : x.shape[1]]
                attention_mask = attention_mask[:, : x.shape[1]]

            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids      = label_ids.to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                continue

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            sub_losses += loss.item()
            sub_count  += 1

        if sub_count > 0:
            total += sub_losses / sub_count
            count += 1
    return total / max(count, 1)
