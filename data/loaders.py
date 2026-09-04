"""Data loading, normalization, and PyTorch dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_datasets(cfg_data: dict) -> dict[str, pd.DataFrame]:
    """Load telemetry and split info from the AmLight-SNMP-TS dataset.

    This repository does not include the dataset itself — it must be
    obtained separately (see the README's Dataset section) and placed at
    `cfg_data["raw_dir"]` (default: "data/AmLight-SNMP-TS/"), containing at
    minimum `telemetry.parquet` and, optionally, `split_info.parquet`.

    Returns dict with keys:
        "telemetry"       : wide-format DataFrame (full dataset, resampled if configured)
        "telemetry_clean" : same
        "split_info"      : DataFrame with train/val/test boundaries, or None
    """
    raw_dir = Path(cfg_data.get("raw_dir", "data/AmLight-SNMP-TS"))
    parquet = raw_dir / "telemetry.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Dataset not found at {parquet}.\n"
            f"AmLight-SNMP-TS is published separately from this code repository — "
            f"see the README's Dataset section for how to obtain it, then place "
            f"it at {raw_dir}/ (or set data.raw_dir in config.yaml)."
        )
    print(f"  Loading telemetry from {parquet} …")
    telemetry = pd.read_parquet(parquet)

    freq = cfg_data.get("resample_freq")
    if freq:
        telemetry = resample_telemetry(telemetry, freq)
        print(f"  Resampled to {freq}: {len(telemetry):,} timesteps")

    split_path = raw_dir / "split_info.parquet"
    if split_path.exists():
        split_info = pd.read_parquet(split_path)
        print(f"  Split info loaded from {split_path}")
        for _, row in split_info.iterrows():
            print(f"    {row['split']:6s}: {row['start']} → {row['end']}")
    else:
        split_info = None
        print(f"  [INFO] split_info.parquet not found — splits computed from fractions.")

    return {"telemetry": telemetry, "telemetry_clean": telemetry, "split_info": split_info}


def resample_telemetry(telemetry: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Resample 1-min telemetry to coarser resolution.

    util_norm and asym columns use mean aggregation.
    in_gbps / out_gbps use mean (sustained throughput).
    The DatetimeIndex is preserved after resampling.
    """
    return telemetry.resample(freq).mean()


def get_active_link_indices(
    telemetry: pd.DataFrame,
    threshold: float = 0.005,
) -> list[int]:
    """Return indices of links whose mean util_norm exceeds threshold.

    Pass the FULL telemetry DataFrame (not just a split) so that links that
    become active only in later periods are not excluded. Selecting which links
    exist is not data leakage — no prediction labels are used.

    Parameters
    ----------
    threshold : links with mean util < threshold are considered idle and excluded.
                Default 0.005 (0.5%) keeps ~32 of 195 AmLight links at 60-min.
    """
    util_cols = sorted(
        [c for c in telemetry.columns if c.endswith("_util_norm")],
        key=lambda c: int(c.split("_")[1]),
    )
    n_links = len(util_cols)
    active = []
    for pos, col in enumerate(util_cols):
        if float(telemetry[col].mean()) >= threshold:
            active.append(pos)
    print(f"  Active links (mean util ≥ {threshold:.1%}): {len(active)} / {n_links}")
    return active


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_util_matrix(
    telemetry: pd.DataFrame,
    capacity_gbps: float = 100.0,  # ignored when util_norm columns are present
) -> np.ndarray:
    """Return per-link utilization normalized to [0, 1].  Shape: (n_steps, n_links).

    Reads pre-computed util_norm columns (link_{i:03d}_util_norm) from the dataset,
    which already incorporate per-link interface speed. The capacity_gbps argument
    is kept for API compatibility but is ignored when util_norm columns exist.
    """
    cols = sorted(c for c in telemetry.columns if c.endswith("_util_norm"))
    return np.stack([telemetry[c].values for c in cols], axis=1).astype(np.float32)


def get_asym_matrix(telemetry: pd.DataFrame) -> np.ndarray:
    """Return traffic asymmetry per link. Shape: (n_steps, n_links). Range [-1, +1].

    asym = (out - in) / (|out| + |in| + ε)
      +1 → fully outbound-dominant
      -1 → fully inbound-dominant
       0 → symmetric
    """
    cols = sorted(c for c in telemetry.columns if c.endswith("_asym"))
    return np.stack([telemetry[c].values for c in cols], axis=1).astype(np.float32)


def get_util_df(
    telemetry: pd.DataFrame,
    capacity_gbps: float = 100.0,  # ignored — see get_util_matrix
) -> pd.DataFrame:
    """Return per-link utilization (normalized) as a DataFrame."""
    util_cols = sorted(c for c in telemetry.columns if c.endswith("_util_norm"))
    data = {c.replace("_util_norm", "_util"): telemetry[c].values for c in util_cols}
    return pd.DataFrame(data, index=telemetry.index)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def load_all_splits(
    telemetry: pd.DataFrame,
    split_info: pd.DataFrame | None = None,
    domain_adapt_frac: float = 0.65,
    train_frac: float = 0.10,
    val_frac: float = 0.05,
    test_frac: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """Return all four temporal partitions as a dict.

    Keys: "domain_adapt", "train", "val", "test"

    When split_info is provided (part of the AmLight-SNMP-TS dataset release),
    uses the persisted timestamp boundaries so every experiment uses identical splits.
    Falls back to fraction-based splitting when split_info is absent.

    Usage in experiments:
        splits     = load_all_splits(telemetry, datasets.get("split_info"))
        da_tel     = splits["domain_adapt"]   # Stage 1 — self-supervised adaptation
        train_tel  = splits["train"]          # Stage 2 — few-shot task training
        val_tel    = splits["val"]            # early stopping / model selection
        test_tel   = splits["test"]           # final held-out evaluation
    """
    if split_info is not None:
        result: dict[str, pd.DataFrame] = {}
        for name in ("domain_adapt", "train", "val", "test"):
            rows = split_info[split_info["split"] == name]
            if rows.empty:
                continue
            row  = rows.iloc[0]
            mask = (telemetry.index >= row["start"]) & (telemetry.index <= row["end"])
            result[name] = telemetry.loc[mask]
        return result

    n    = len(telemetry)
    n_da = int(n * domain_adapt_frac)
    n_tr = int(n * train_frac)
    n_vl = int(n * val_frac)
    n_te = n - n_da - n_tr - n_vl
    return {
        "domain_adapt": telemetry.iloc[:n_da],
        "train":        telemetry.iloc[n_da : n_da + n_tr],
        "val":          telemetry.iloc[n_da + n_tr : n_da + n_tr + n_vl],
        "test":         telemetry.iloc[n_da + n_tr + n_vl :],
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """Per-column min-max normalization fitted on training data only."""

    def __init__(self) -> None:
        self.min_: Optional[pd.Series] = None
        self.max_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        self.min_ = df.min()
        self.max_ = df.max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.min_ is not None, "call fit() first"
        scale = (self.max_ - self.min_).replace(0, 1)
        return (df - self.min_) / scale

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.min_ is not None
        scale = (self.max_ - self.min_).replace(0, 1)
        return df * scale + self.min_

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Sliding-window dataset for TSFM training and evaluation.

    Each sample is a (context, target) pair:
        context : float32 tensor of shape (context_len, n_features)
        target  : float32 tensor of shape (prediction_len, n_features)
    """

    def __init__(
        self,
        data: np.ndarray,
        context_len: int,
        prediction_len: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data = torch.from_numpy(data.astype(np.float32))
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.stride = stride
        total = context_len + prediction_len
        n_all = max(0, (len(data) - total) // stride + 1)
        self.indices = list(range(n_all))
        self.n_samples = len(self.indices)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx] * self.stride
        ctx_end = start + self.context_len
        tgt_end = ctx_end + self.prediction_len
        return self.data[start:ctx_end], self.data[ctx_end:tgt_end]


class MaskedWindowDataset(Dataset):
    """WindowDataset wrapper that applies temporal masking as augmentation.

    Used in Stage 1 self-supervised adaptation (exp3) for Chronos, Moirai,
    and TimesFM. Masking forces the model to infer missing signal from context.
    Masking types applied per sample:
      - temporal: individual random time steps zeroed
      - span: contiguous blocks of steps zeroed
      - feature: one random link channel zeroed (30% of samples)
    """

    def __init__(
        self,
        base_dataset: WindowDataset,
        mask_ratio: float = 0.15,
        span_mask_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.mask_ratio = mask_ratio
        self.span_mask_ratio = span_mask_ratio
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        context, target = self.base[idx]
        ctx = context.clone()
        T, F = ctx.shape

        # Random temporal masking
        rand_mask = torch.from_numpy(
            (self.rng.random(T) < self.mask_ratio).astype(np.float32)
        ).unsqueeze(1)
        ctx = ctx * (1.0 - rand_mask)

        # Contiguous span masking (one span per sample)
        span_len = max(1, int(T * self.span_mask_ratio))
        span_start = int(self.rng.integers(0, T - span_len))
        ctx[span_start : span_start + span_len] = 0.0

        # Feature (link) masking: randomly zero one link channel
        if F > 1 and self.rng.random() < 0.3:
            link_idx = int(self.rng.integers(0, F))
            ctx[:, link_idx] = 0.0

        return ctx, target


class PatchMaskedWindowDataset(Dataset):
    """WindowDataset wrapper for PatchTST self-supervised pre-training.

    Implements the masked patch prediction objective from Nie et al. (ICLR 2023):
      - Context is split into non-overlapping patches of `patch_len` timesteps
      - 40% of patches are randomly zeroed out (BERT-style: kept at original
        position with positional encoding, not dropped)
      - Target is the original (unmasked) context — the model reconstructs
        masked patches from visible ones
      - Loss is computed only on masked patch positions

    Returns (masked_context, original_context, patch_mask) where:
      masked_context  : (context_len, n_features) — input with zeroed patches
      original_context: (context_len, n_features) — reconstruction target
      patch_mask      : (n_patches,) bool — True = patch was masked
    """

    def __init__(
        self,
        base_dataset: WindowDataset,
        patch_len: int = 16,
        mask_ratio: float = 0.40,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(seed)
        ctx_len = base_dataset.context_len
        self.n_patches = ctx_len // patch_len

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, _ = self.base[idx]          # target unused — reconstruction task
        original = context.clone()
        masked   = context.clone()

        # Sample which patches to mask (uniformly at random, no contiguous bias)
        n_mask = max(1, int(self.n_patches * self.mask_ratio))
        mask_idx = self.rng.choice(self.n_patches, size=n_mask, replace=False)
        patch_mask = torch.zeros(self.n_patches, dtype=torch.bool)
        for p in mask_idx:
            start = p * self.patch_len
            masked[start : start + self.patch_len] = 0.0
            patch_mask[p] = True

        return masked, original, patch_mask


def build_window_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_len: int,
    prediction_len: int,
    capacity_gbps: float = 100.0,
    masked: bool = False,
    stride: int = 144,
) -> tuple[WindowDataset, WindowDataset, WindowDataset, Normalizer]:
    """Normalize util and wrap DataFrames into WindowDatasets.

    Uses max(in_util, out_util) per link as the training signal for Stage 1.

    stride=144 (2.4-hour spacing at 1-min resolution) gives ~2500 windows/year.
    stride=1 gives ~360k windows — slower, diminishing returns from high overlap.
    """
    train_util = get_util_df(train_df, capacity_gbps)
    val_util   = get_util_df(val_df,   capacity_gbps)
    test_util  = get_util_df(test_df,  capacity_gbps)

    norm = Normalizer().fit(train_util)
    train_norm = norm.transform(train_util).values
    val_norm   = norm.transform(val_util).values
    test_norm  = norm.transform(test_util).values

    train_ds = WindowDataset(train_norm, context_len, prediction_len, stride=stride)
    val_ds   = WindowDataset(val_norm,   context_len, prediction_len, stride=stride)
    test_ds  = WindowDataset(test_norm,  context_len, prediction_len, stride=stride)

    if masked:
        train_ds = MaskedWindowDataset(train_ds)

    return train_ds, val_ds, test_ds, norm
