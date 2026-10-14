"""Models available to the standalone EEG trainer."""

from .ADformer import ADformer
from .CodimNet import CodimNet
from .EEGNet import EEGNet
from .models1.DBConformer import DBConformer

__all__ = ["ADformer", "CodimNet", "EEGNet", "DBConformer", "get_model"]

_MODELS = {
    "ADformer": ADformer,
    "CodimNet": CodimNet,
    "EEGNet": EEGNet,
    "DBConformer": DBConformer,
}


def get_model(model_name, configs):
    """Build a model using the dataset's channels, length, and class count."""
    if model_name not in _MODELS:
        available = ", ".join(_MODELS)
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    return _MODELS[model_name](configs=configs)
