"""TimesFM (Google) wrapper — supports 1.x and 2.5+ API.

TimesFM is a decoder-only foundation model for time series forecasting,
pre-trained on 100 billion time points.

1.x API: timesfm.TimesFmHparams, timesfm.TimesFm, timesfm.TimesFmCheckpoint
2.5 API: timesfm.TimesFM_2p5_200M_torch, timesfm.ForecastConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import TSFMBase


class TimesFMModel(TSFMBase):

    name = "timesfm"
    _hf_model_id = "google/timesfm-1.0-200m-pytorch"

    def __init__(
        self,
        context_len: int,
        prediction_len: int,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(context_len, prediction_len, device)
        self._model = None
        self._hparams = None
        self._api_version: str | None = None

    def load(self) -> None:
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "Install TimesFM: pip install timesfm\n"
                "See https://github.com/google-research/timesfm"
            )

        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID,
                torch_compile=False,
            )
            self._model.compile(timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.prediction_len,
                per_core_batch_size=32,
            ))
            self._api_version = "2.5"
        elif hasattr(timesfm, "TimesFmHparams"):
            # API 1.x supports max context of 512 tokens; cap silently if larger.
            _ctx_1x = min(self.context_len, 512)
            if _ctx_1x < self.context_len:
                print(f"  [timesfm] API 1.x: context capped at {_ctx_1x} (model max); "
                      f"upgrade to 2.5 for full {self.context_len}-step context.")
            # PatchedTimeSeriesDecoder requires context_len % patch_len == 0 (patch_len=32).
            # Round up to next multiple of 32; forecast() will left-pad the context to match.
            _PATCH = 32
            _ctx_padded = ((_ctx_1x + _PATCH - 1) // _PATCH) * _PATCH
            if _ctx_padded != _ctx_1x:
                print(f"  [timesfm] API 1.x: context_len {_ctx_1x} → {_ctx_padded} "
                      f"(padded to multiple of patch_len={_PATCH})")
            self._ctx_1x = _ctx_padded
            self._hparams = timesfm.TimesFmHparams(
                backend="gpu" if "cuda" in self.device else "cpu",
                per_core_batch_size=32,
                horizon_len=self.prediction_len,
                context_len=_ctx_padded,
            )
            self._model = timesfm.TimesFm(
                hparams=self._hparams,
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self._hf_model_id
                ),
            )
            self._api_version = "1.x"
        else:
            raise RuntimeError(
                f"Unknown timesfm API. Available: "
                f"{[a for a in dir(timesfm) if not a.startswith('_')]}"
            )

        self._loaded = True

    def forecast(self, context: np.ndarray) -> np.ndarray:
        assert self._loaded, "call load() first"
        ctx = context.flatten()[-self.context_len:].astype(np.float32)
        if self._api_version == "2.5":
            out = self._model.forecast(inputs=[ctx], horizon=self.prediction_len)
            point_forecast = out[0] if isinstance(out, (tuple, list)) else out
        else:
            # Left-pad to match hparams context_len (rounded up to multiple of patch_len=32)
            target_len = getattr(self, "_ctx_1x", len(ctx))
            if len(ctx) < target_len:
                ctx = np.pad(ctx, (target_len - len(ctx), 0), mode="edge")
            point_forecast, _ = self._model.forecast(inputs=[ctx], freq=[0])
        arr = np.asarray(point_forecast)
        return arr[0] if arr.ndim > 1 else arr

    def _get_torch_module(self) -> torch.nn.Module:
        if self._api_version == "2.5":
            if isinstance(self._model, torch.nn.Module):
                return self._model
            for attr in ("model", "_model", "backbone"):
                m = getattr(self._model, attr, None)
                if isinstance(m, torch.nn.Module):
                    return m
            raise RuntimeError("Cannot find torch.nn.Module inside timesfm 2.5 model")
        return self._model._model

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
        """Fine-tune TimesFM using its PyTorch backend.

        API 1.x  (timesfm.TimesFm + HF PyTorch checkpoint):
          The inner PatchedTimeSeriesDecoder supports standard gradient training
          via forward(input_ts, input_padding, freq). This is the recommended path.

        API 2.5  (TimesFM_2p5_200M_torch):
          The inner model forward() may require (x, freq) positional args. We try
          two calling conventions and raise a clear error if neither works — do NOT
          silently return zero-shot weights.
        """
        assert self._loaded, "call load() first"
        Path(weights_dir).mkdir(parents=True, exist_ok=True)

        torch_model = self._get_torch_module()

        if self._api_version == "1.x":
            # PatchedTimeSeriesDecoder: forward(input_ts, input_padding, freq)
            # freq shape must be (1,) not (B,) — see _run_epoch_1x for details.
            _epoch_fn = _run_epoch_1x
            print(f"  [timesfm] API 1.x: using PatchedTimeSeriesDecoder training loop")
        else:
            # API 2.5: try (x,) then (x, freq) calling convention.
            # Do NOT fall back silently — raise if neither works so the issue is visible.
            _epoch_fn = _resolve_epoch_fn_2p5(torch_model, self.context_len, self.device)

        torch_model = torch_model.to(self.device)
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            torch_model.train()
            train_loss = _epoch_fn(torch_model, train_loader, optimizer, criterion, self.device)
            torch_model.eval()
            val_loss = _epoch_fn(torch_model, val_loader, None, criterion, self.device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"  [{self.name}] epoch {epoch+1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(torch_model.state_dict(), Path(weights_dir) / f"{self.name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{self.name}] early stopping at epoch {epoch+1}")
                    break

        torch_model.load_state_dict(
            torch.load(Path(weights_dir) / f"{self.name}_best.pt", map_location=self.device)
        )
        return history

    def save_weights(self, path: str | Path) -> None:
        assert self._loaded
        try:
            torch.save(self._get_torch_module().state_dict(), path)
        except RuntimeError as e:
            print(f"  [timesfm] save_weights skipped ({e})")

    def load_weights(self, path: str | Path) -> None:
        assert self._loaded
        self._get_torch_module().load_state_dict(
            torch.load(path, map_location=self.device)
        )


def _revin_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Instance normalization (RevIN-style, no learnable affine).

    Matches TimesFM 1.0 pre-training: normalize each series by mean and std
    computed over the context window before feeding to the model.
    Returns (normalized_x, mean, std) for denormalization of predictions.
    """
    mean = x.mean(dim=1, keepdim=True)                        # (B, 1)
    std  = x.std(dim=1, keepdim=True).clamp(min=1e-5)        # (B, 1)
    return (x - mean) / std, mean, std


def _revin_denorm(pred: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reverse RevIN normalization on model output."""
    return pred * std + mean


def _run_epoch(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    """Generic epoch with RevIN normalization (TimesFM pre-training objective)."""
    total, count = 0.0, 0
    for context, target in loader:
        B, ctx_len, n_links = context.shape
        link_idx = torch.randint(0, n_links, (B,))
        x = context[torch.arange(B), :, link_idx].to(device)
        y = target[torch.arange(B), :, link_idx].to(device)
        x_norm, mean, std = _revin_norm(x)
        pred = model(x_norm)
        if hasattr(pred, "logits"):
            pred = pred.logits
        t    = min(pred.shape[1], y.shape[1])
        pred = _revin_denorm(pred[:, :t], mean, std)
        loss = criterion(pred, y[:, :t])
        if not torch.isfinite(loss):
            continue
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def _run_epoch_1x(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    """Training epoch for TimesFM 1.x PatchedTimeSeriesDecoder.

    The decoder requires (input_ts, input_padding, freq) where:
      input_padding: bool mask, True = padded position (all False = no padding)
      freq: integer frequency class (0 = high-frequency / sub-hourly)
    Output is (mean_predictions, full_predictions); we use the mean for MSE loss.
    Channel-independent: one random link sampled per window (same as Moirai).
    Processing all 173 links per window at once (B * n_links sequences) exceeds
    A30 VRAM; random sampling provides equivalent coverage over many epochs.
    """
    _PATCH = 32  # PatchedTimeSeriesDecoder patch_len; ctx_len must be a multiple
    total, count = 0.0, 0
    for context, target in loader:
        B, ctx_len, n_links = context.shape
        link_idx = torch.randint(0, n_links, (B,))
        x = context[torch.arange(B), :, link_idx].to(device)   # (B, ctx_len)
        y = target[torch.arange(B), :, link_idx].to(device)    # (B, pred_len)
        # Left-pad to next multiple of patch_len so view(B, -1, 32) succeeds
        if ctx_len % _PATCH != 0:
            pad_len = _PATCH - (ctx_len % _PATCH)
            x = torch.nn.functional.pad(x, (pad_len, 0), mode="replicate")
        x_norm, mean, std = _revin_norm(x)
        pad = torch.zeros_like(x_norm, dtype=torch.float32)      # no padding
        frq = torch.zeros(1, dtype=torch.long, device=device)    # freq=0 (sub-hourly/1-min), shape (1,) not (B,)

        out  = model(x_norm, pad, frq)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        # Output shape variants:
        #   4-D: (B, num_decode_patches, out_per_patch, num_quantiles)
        #   3-D: (B, num_decode_patches, out_per_patch) or (B, seq_len, num_quantiles)
        #   2-D: (B, total_steps)
        # Assemble all decode patches (not just the last one) so we have as many
        # output steps as the model can produce.  With prediction_len > patch_size
        # (e.g. 192 > 128) taking only the last patch would drop half the horizon.
        if pred.ndim == 4:
            pred = pred.mean(dim=-1)                    # (B, n_dp, opp) — avg quantiles
            pred = pred.reshape(pred.shape[0], -1)      # (B, n_dp * opp)
        elif pred.ndim == 3:
            pred = pred.mean(dim=-1)                    # (B, seq_len)
        # Compute loss on available overlap: model may output fewer steps than pred_len
        # (TimesFM 1.x native patch size is 128; training forward() may produce 1 patch).
        t    = min(pred.shape[1], y.shape[1])
        pred = _revin_denorm(pred[:, :t], mean, std)

        loss = criterion(pred, y[:, :t])
        if not torch.isfinite(loss):
            continue
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def _run_epoch_2p5(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    """Training epoch for TimesFM 2.5 inner model.

    TimesFM 2.5 inner model forward: (input_ts, freq) → (B, pred_len) predictions.
    freq=0 means sub-hourly (correct for 1-minute SNMP data).
    Channel-independent: each link processed as a separate univariate series.
    """
    total, count = 0.0, 0
    for context, target in loader:
        B, ctx_len, n_links = context.shape
        x   = context.permute(0, 2, 1).reshape(B * n_links, ctx_len).to(device)
        y   = target.permute(0, 2, 1).reshape(B * n_links, target.shape[1]).to(device)
        x_norm, mean, std = _revin_norm(x)
        frq = torch.zeros(B * n_links, dtype=torch.long, device=device)  # freq class 0 (sub-hourly/1-min)

        out  = model(x_norm, frq)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(pred, "logits"):
            pred = pred.logits
        if pred.ndim > 2:
            pred = pred.reshape(B * n_links, -1)
        t    = min(pred.shape[1], y.shape[1])
        pred = _revin_denorm(pred[:, :t], mean, std)
        loss = criterion(pred, y[:, :t])
        if not torch.isfinite(loss):
            continue
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def _resolve_epoch_fn_2p5(torch_model: torch.nn.Module, context_len: int, device: str):
    """Determine the correct epoch function for TimesFM 2.5 by testing calling conventions.

    Tries (x, freq) first, then (x,). Raises RuntimeError if neither works so the
    problem is immediately visible — never silently falls back to zero-shot weights.
    """
    x   = torch.zeros(1, context_len, device=device)
    frq = torch.zeros(1, dtype=torch.long, device=device)

    # Convention 1: forward(x, freq) — TimesFM 2.5 standard
    try:
        with torch.no_grad():
            out = torch_model(x, frq)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(pred, torch.Tensor) and pred.numel() > 0:
            print("  [timesfm] API 2.5: forward(x, freq) convention confirmed")
            return _run_epoch_2p5
    except Exception:
        pass

    # Convention 2: forward(x) — some 2.5 variants expose simplified forward
    try:
        with torch.no_grad():
            out = torch_model(x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(pred, torch.Tensor) and pred.numel() > 0:
            print("  [timesfm] API 2.5: forward(x) convention confirmed")
            return _run_epoch
    except Exception:
        pass

    raise RuntimeError(
        "TimesFM 2.5: cannot find a compatible forward() for gradient training.\n"
        "Tried: forward(x, freq) and forward(x).\n"
        "Install timesfm 1.x for a stable training path:\n"
        "  pip install 'timesfm[torch]==1.0.0'"
    )
