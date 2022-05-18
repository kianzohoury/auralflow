from . architectures import (
    SpectrogramNetSimple, SpectrogramNetLSTM, SpectrogramNetVAE
)
from . base import SeparationModel
from . mask_model import SpectrogramMaskModel


__all__ = [
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE",
    "SeparationModel",
    "SpectrogramMaskModel"
]


def create_model(configuration: dict) -> SeparationModel:
    """Creates a new instance of a model with its given configuration."""
    separation_task = configuration["model_params"]["separation_task"]
    if separation_task == "mask":
        base_class = SpectrogramMaskModel
    else:
        base_class = lambda x: x
        pass

    model = base_class(configuration)
    return model
