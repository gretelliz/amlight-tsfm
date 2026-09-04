"""IBM Granite Tiny Time Mixer (TTM) wrapper.

Model card: ibm-granite/granite-timeseries-ttm-r2
Install:    pip install tsfm-public

TTM is a lightweight MLP-Mixer model (~1M params) pre-trained on standard
benchmark datasets at context_length=512, prediction_length=96h.

For our 36h horizon:
  - Zero-shot: load pre-trained 96h model, slice output to prediction_len.
  - Stage 1 domain adaptation (SSL): fine-tune on AmLight utilization windows
    using TTM's native objective — RevIN-normalized next-horizon MSE against
    the actual future trajectory (self-supervised, no task labels).
    We compute MSE only on the first prediction_len steps of the 96h output,
    adapting the full model to AmLight temporal dynamics.

This differs from PatchTST (masked patch reconstruction) and Chronos (tokenized NLL):
TTM learns by predicting the next 36h of utilization from the observed 512h context,
with per-sample instance normalization removing scale bias before the loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import TSFMBase

_HF_REPO = "ibm-granite/granite-timeseries-ttm-r2"
_PRETRAINED_HORIZON  = 96   # TTM-R2 native prediction length (steps)
_PRETRAINED_CONTEXT  = 512  # TTM-R2 native context length (steps)


def _require_tsfm() -> None:
    try:
        import tsfm_public  # noqa: F401
    except ImportError:
        raise ImportError(
            "TTM requires tsfm-public. Install with:\n"
            "  pip install tsfm-public\n"
            "or:\n"
            "  pip install \"tsfm_public@git+https://github.com/ibm-granite/granite-tsfm.git\""
        )


class TTMModel(TSFMBase):
    """IBM Granite Tiny Time Mixer — zero-shot and domain-adapted.

    TTM-R2 was pre-trained with context_length=512 steps and prediction_length=96
    steps on hourly benchmark datasets. At AmLight's 1-minute resolution this
    corresponds to ~8.5h of context and a 96-minute forecast horizon.

    Because TTM's MLP-Mixer layers are sized for exactly 512/patch_len patches,
    we always feed the last 512 steps of context regardless of the global
    context_len in config. Evaluation targets are also computed over the native
    96-step horizon so the task remains internally consistent.
    """

    name = "ttm"
    native_context_len = _PRETRAINED_CONTEXT   # 512 — never changes
    native_pred_len    = _PRETRAINED_HORIZON    # 96  — never changes

    def __init__(
        self,
        context_len: int,
        prediction_len: int,
        device: Optional[str] = None,
        model_size: str = "r2",   # kept for API parity; TTM-R2 is the default
    ) -> None:
        # Store native sizes; ignore the global context/pred lengths for inference
        super().__init__(self.native_context_len, self.native_pred_len, device)
        self.model_size = model_size
        self._inner = None

    def load(self) -> None:
        """Download pre-trained TTM-R2 from HuggingFace."""
        _require_tsfm()
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        # Load with native architecture — do NOT override context_length or
        # prediction_length, which would break the pre-trained MLP weight shapes.
        self._inner = TinyTimeMixerForPrediction.from_pretrained(
            _HF_REPO,
            num_input_channels=1,
        ).to(self.device)
        self._inner.eval()
        self._loaded = True

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def forecast(self, context: np.ndarray) -> np.ndarray:
        """Single-window forecast. context: (L,) or (L, 1) → (prediction_len,)."""
        assert self._loaded, "call load() first"
        ctx = context.flatten()[-self.context_len:]
        # Pad if shorter than context_len
        if len(ctx) < self.context_len:
            pad = np.zeros(self.context_len - len(ctx), dtype=np.float32)
            ctx = np.concatenate([pad, ctx])
        x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            out = self._inner(past_values=x)
        # prediction_outputs: (1, 96, 1) → slice to prediction_len
        return out.prediction_outputs[0, : self.prediction_len, 0].cpu().numpy()

    def forecast_many(self, contexts: list[np.ndarray], batch_size: int = 64) -> list[np.ndarray]:
        """Batched inference. Each context: (L,) or (L, 1) → (prediction_len,)."""
        assert self._loaded
        results: list[np.ndarray] = []
        self._inner.eval()

        for i in range(0, len(contexts), batch_size):
            batch_np = np.stack([
                np.concatenate([
                    np.zeros(max(0, self.context_len - len(c.flatten())), dtype=np.float32),
                    c.flatten()[-self.context_len:],
                ])
                for c in contexts[i : i + batch_size]
            ])  # (B, context_len)
            x = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                out = self._inner(past_values=x)
            preds = out.prediction_outputs[:, : self.prediction_len, 0].cpu().numpy()
            results.extend(list(preds))
        return results

    # -----------------------------------------------------------------------
    # Stage 1 domain adaptation — TTM's native SSL objective
    # -----------------------------------------------------------------------

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
        """Stage 1 domain adaptation using TTM's native SSL objective.

        TTM is adapted on AmLight utilization via RevIN-normalized next-horizon
        MSE forecasting — the same self-supervised objective used in its original
        pre-training. No task labels are required; targets are derived directly
        from the raw telemetry (the next prediction_len timesteps following each
        context window).

        Training uses the full model (backbone + prediction head). MSE is
        computed on the first prediction_len steps of the 96h output, aligning
        the model to AmLight's 36h operational horizon without replacing the
        pre-trained prediction head.

        Channel-independent processing: each batch of B windows over n_links
        channels is reshaped to B*n_links univariate samples, expanding training
        data by n_links without architecture changes.
        """
        assert self._loaded, "call load() first"
        Path(weights_dir).mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(
            self._inner.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            self._inner.train()
            train_loss = _epoch_loss_ttm(
                self._inner, train_loader, optimizer, self.device, self.prediction_len
            )
            self._inner.eval()
            val_loss = _epoch_loss_ttm(
                self._inner, val_loader, None, self.device, self.prediction_len
            )
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(
                f"  [{self.name}] epoch {epoch+1:3d}/{epochs}"
                f"  train={train_loss:.4f}  val={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(self._inner.state_dict(), Path(weights_dir) / f"{self.name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{self.name}] early stopping at epoch {epoch + 1}")
                    break

        best_path = Path(weights_dir) / f"{self.name}_best.pt"
        if best_path.exists():
            self._inner.load_state_dict(torch.load(best_path, map_location=self.device))
        return history

    # -----------------------------------------------------------------------
    # Weight persistence
    # -----------------------------------------------------------------------

    def save_weights(self, path: str | Path) -> None:
        assert self._loaded
        torch.save(self._inner.state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        assert self._loaded, "call load() before load_weights()"
        self._inner.load_state_dict(torch.load(path, map_location=self.device))


# ---------------------------------------------------------------------------
# TTM training epoch
# ---------------------------------------------------------------------------

def _epoch_loss_ttm(
    model,
    loader: DataLoader,
    optimizer,
    device: str,
    prediction_len: int,
) -> float:
    """One training/validation epoch for TTM Stage 1 SSL.

    Input:  context (B, L, n_links) — masked time series windows
    Target: future  (B, H, n_links) — next prediction_len steps

    Channel-independent: reshape to (B*n_links, L, 1) and (B*n_links, H, 1).
    Loss: MSE on the first prediction_len steps of the 96h TTM output.
    RevIN instance normalization is applied by TTM internally.
    """
    total, count = 0.0, 0

    for context, target in loader:
        B, ctx_len, n_links = context.shape

        # Channel-independent reshape
        x = context.permute(0, 2, 1).reshape(B * n_links, ctx_len, 1).float().to(device)
        y = target.permute(0, 2, 1).reshape(B * n_links, target.shape[1], 1).float().to(device)

        # TTM forward: predict _PRETRAINED_HORIZON (96h) steps
        out = model(past_values=x)
        # prediction_outputs: (B*n_links, 96, 1)

        # Align to the smaller of TTM's native horizon and the dataloader's target length
        horizon = min(prediction_len, y.shape[1])
        pred = out.prediction_outputs[:, :horizon, :]
        tgt  = y[:, :horizon, :]

        loss = F.mse_loss(pred, tgt)

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
