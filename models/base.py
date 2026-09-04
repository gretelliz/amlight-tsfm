"""Abstract base class for all TSFM and baseline models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


class TSFMBase(ABC):
    """Common interface for all time series models used in this study.

    Subclasses implement load(), forecast(), and finetune().
    """

    name: str = "base"

    def __init__(self, context_len: int, prediction_len: int, device: Optional[str] = None) -> None:
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Download and load pre-trained weights."""

    @abstractmethod
    def forecast(self, context: np.ndarray) -> np.ndarray:
        """Forecast future values from a context window.

        Parameters
        ----------
        context : ndarray of shape (context_len,) or (context_len, n_features)

        Returns
        -------
        ndarray of shape (prediction_len,) or (prediction_len, n_features)
        """

    def forecast_many(self, contexts: list[np.ndarray]) -> list[np.ndarray]:
        """Forecast for a list of contexts. Override in subclasses for batched GPU efficiency."""
        return [self.forecast(c) for c in contexts]

    def forecast_upper(self, context: np.ndarray, q: float = 0.9) -> np.ndarray:
        """q-th quantile forecast. Probabilistic models override to expose their
        predictive distribution; deterministic models fall back to point forecast.
        """
        return self.forecast(context)

    def forecast_samples_many(
        self,
        contexts: list[np.ndarray],
        **kwargs,
    ) -> list[np.ndarray]:
        """Return raw samples for each context. Shape per item: (n_samples, pred_len).

        Probabilistic models (Chronos, Moirai) override to return n_samples > 1,
        enabling calibrated P(max > threshold) estimates for T1 exceedance.
        This base implementation wraps forecast_many() with n_samples=1 (hard prediction).
        """
        point_forecasts = self.forecast_many(contexts)
        return [
            np.asarray(f).flatten()[: self.prediction_len][None, :]
            for f in point_forecasts
        ]

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
        """Fine-tune the model on domain data.

        Returns a history dict with 'train_loss' and 'val_loss' lists.
        Subclasses may override for model-specific fine-tuning strategies.
        """
        raise NotImplementedError(f"{self.name} does not support fine-tuning via this interface.")

    def save_weights(self, path: str | Path) -> None:
        raise NotImplementedError

    def load_weights(self, path: str | Path) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(context={self.context_len}, pred={self.prediction_len})"
