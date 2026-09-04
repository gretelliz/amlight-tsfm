from .base import TSFMBase
from .chronos import ChronosModel
from .moirai import MoiraiModel
from .patchtst import PatchTSTModel
from .timesfm import TimesFMModel
from .ttm import TTMModel

MODELS: dict[str, type[TSFMBase]] = {
    "chronos": ChronosModel,
    "moirai": MoiraiModel,
    "timesfm": TimesFMModel,
    "patchtst": PatchTSTModel,
    "ttm": TTMModel,
}

__all__ = [
    "TSFMBase", "ChronosModel", "MoiraiModel", "TimesFMModel",
    "PatchTSTModel", "TTMModel", "MODELS",
]
