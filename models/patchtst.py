"""PatchTST — domain-specific baseline trained from scratch on AmLight data.

PatchTST divides time series into non-overlapping patches (sub-sequences) and
processes them with a standard Transformer encoder. Unlike the three foundation
models, PatchTST has no pre-training benefit; it learns only from AmLight data.
Its role is to bound the performance achievable with domain-specific training
alone, so we can quantify the added value of large-scale pre-training.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers", ICLR 2023.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import TSFMBase


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class _PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int, n_features: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        n_patches = T // self.patch_len
        x = x[:, : n_patches * self.patch_len, :]
        x = x.reshape(B, n_patches, self.patch_len * C)
        return self.norm(self.proj(x))  # (B, n_patches, d_model)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchTSTModule(nn.Module):
    """PatchTST with two operating modes:

    Forecasting (default): context → future H-step trajectory.
    Pre-training: masked context → reconstruct masked patches (Nie et al. 2023).
    """

    def __init__(
        self,
        context_len: int,
        prediction_len: int,
        n_features: int = 1,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.n_patches = context_len // patch_len

        self.embedding = _PatchEmbedding(patch_len, d_model, n_features)
        self.pos_enc = _PositionalEncoding(d_model, max_len=self.n_patches + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Forecasting head: flatten all patch encodings → future trajectory
        self.head = nn.Linear(d_model * self.n_patches, prediction_len * n_features)
        # Reconstruction head: per-patch linear → patch_len values (pre-training only)
        self.recon_head = nn.Linear(d_model, patch_len * n_features)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: (B, T, C) → (B, n_patches, d_model)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        emb = self.embedding(x)
        emb = self.pos_enc(emb)
        return self.encoder(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecasting forward. Returns (B, prediction_len, n_features)."""
        B = x.shape[0]
        enc = self._encode(x)                          # (B, n_patches, d_model)
        out = self.head(enc.flatten(1))                # (B, pred_len * C)
        return out.reshape(B, self.prediction_len, self.n_features)

    def forward_pretrain(
        self, x: torch.Tensor, patch_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pre-training forward: reconstruct masked patches.

        Parameters
        ----------
        x          : (B, T, C) — masked context (zeroed patches)
        patch_mask : (B, n_patches) bool — True = patch was masked

        Returns
        -------
        recon : (B, n_patches, patch_len * n_features) — reconstructed values
                for ALL patches (caller computes loss only on masked positions)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        enc = self._encode(x)                          # (B, n_patches, d_model)
        return self.recon_head(enc)                    # (B, n_patches, patch_len*C)


# ---------------------------------------------------------------------------
# TSFMBase wrapper
# ---------------------------------------------------------------------------

class PatchTSTModel(TSFMBase):

    name = "patchtst"

    def __init__(
        self,
        context_len: int,
        prediction_len: int,
        device: Optional[str] = None,
        n_features: int = 1,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(context_len, prediction_len, device)
        self.n_features = n_features
        self._module = PatchTSTModule(
            context_len=context_len,
            prediction_len=prediction_len,
            n_features=n_features,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

    def load(self) -> None:
        # PatchTST has no pre-trained weights to download.
        # Weights are set via finetune() or load_weights().
        self._loaded = True

    def forecast(self, context: np.ndarray) -> np.ndarray:
        assert self._loaded
        ctx = context[-self.context_len:]
        if ctx.ndim == 1:
            ctx = ctx[:, None]
        if len(ctx) < self.context_len:
            pad = np.zeros((self.context_len - len(ctx), ctx.shape[1]), dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=0)
        x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._module(x)
        return out.squeeze(0).cpu().numpy()

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
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(self._module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            self._module.train()
            train_loss = _run_epoch(self._module, train_loader, optimizer, criterion, self.device)
            self._module.eval()
            val_loss = _run_epoch(self._module, val_loader, None, criterion, self.device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"  [{self.name}] epoch {epoch+1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(self._module.state_dict(), Path(weights_dir) / f"{self.name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{self.name}] early stopping at epoch {epoch+1}")
                    break

        self._module.load_state_dict(
            torch.load(Path(weights_dir) / f"{self.name}_best.pt", map_location=self.device)
        )
        return history

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 5,
        weights_dir: str | Path = "models/weights",
    ) -> dict[str, list[float]]:
        """Phase 1: masked patch reconstruction (Nie et al. ICLR 2023).

        Expects DataLoader yielding (masked_ctx, original_ctx, patch_mask) from
        PatchMaskedWindowDataset. Trains the encoder + recon_head; the forecast
        head is not used here. After pretrain(), call finetune() to swap in the
        forecast head with a lower learning rate.
        """
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        # Only train encoder + recon_head during pre-training
        params = list(self._module.encoder.parameters()) + \
                 list(self._module.embedding.parameters()) + \
                 list(self._module.recon_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        history: dict[str, list[float]] = {"pretrain_loss": [], "pretrain_val_loss": []}
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            self._module.train()
            train_loss = _run_epoch_pretrain(self._module, train_loader, optimizer, criterion, self.device)
            self._module.eval()
            val_loss = _run_epoch_pretrain(self._module, val_loader, None, criterion, self.device)
            scheduler.step()

            history["pretrain_loss"].append(train_loss)
            history["pretrain_val_loss"].append(val_loss)
            print(f"  [{self.name}] pretrain {epoch+1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(self._module.state_dict(), Path(weights_dir) / f"{self.name}_pretrained.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{self.name}] pretrain early stopping at epoch {epoch+1}")
                    break

        self._module.load_state_dict(
            torch.load(Path(weights_dir) / f"{self.name}_pretrained.pt", map_location=self.device)
        )
        return history

    def save_weights(self, path: str | Path) -> None:
        torch.save(self._module.state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        self._module.load_state_dict(torch.load(path, map_location=self.device))
        self._loaded = True


def _run_epoch_pretrain(module, loader: DataLoader, optimizer, criterion, device: str) -> float:
    """Pre-training epoch: reconstruct masked patches, loss on masked positions only."""
    total, count = 0.0, 0
    for masked_ctx, original_ctx, patch_mask in loader:
        # masked_ctx:   (B, T, C)
        # original_ctx: (B, T, C)
        # patch_mask:   (B, n_patches) bool
        B, T, C = masked_ctx.shape
        patch_len = module.patch_len
        n_patches = T // patch_len

        # Channel-independent: reshape (B, T, C) → (B*C, T, 1), same as _run_epoch
        x      = masked_ctx.permute(0, 2, 1).reshape(B * C, T, 1).to(device)
        target = original_ctx.permute(0, 2, 1).reshape(B * C, T, 1).to(device)
        # Broadcast patch_mask (B, n_patches) → (B*C, n_patches)
        pmask  = patch_mask.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1).to(device)

        recon = module.forward_pretrain(x, pmask)  # (B*C, n_patches, patch_len)

        # Reconstruct target patches from original context
        tgt_patches = target[:, :n_patches * patch_len, :].reshape(
            B * C, n_patches, patch_len
        )                                          # (B*C, n_patches, patch_len)

        # Loss only on masked positions
        mask_exp = pmask.unsqueeze(-1).expand_as(recon)  # (B*C, n_patches, patch_len)
        if mask_exp.any():
            loss = criterion(recon[mask_exp], tgt_patches[mask_exp])
        else:
            continue

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def _run_epoch(module, loader: DataLoader, optimizer, criterion, device: str) -> float:
    total, count = 0.0, 0
    for context, target in loader:
        B, ctx_len, n_links = context.shape
        # Channel-independent: each link becomes a univariate (T, 1) sample
        x = context.permute(0, 2, 1).reshape(B * n_links, ctx_len, 1).to(device)
        y = target.permute(0, 2, 1).reshape(B * n_links, target.shape[1], 1).to(device)
        pred = module(x)
        loss = criterion(pred, y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)
