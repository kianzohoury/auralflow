import importlib
from typing import Any


def create_model(configuration: dict) -> Any:
    """Creates a new instance of a model with its given configuration."""
    separation_task = configuration["model_params"]["separation_task"]
    if separation_task == "mask":
        base_class = importlib.import_module(
            f"{separation_task}_model", "SpectrogramMaskModel"
        )
    else:
        base_class = lambda x: x
        pass

    model = base_class(configuration)
    return model
