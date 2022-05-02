import importlib


def create_model(configuration: dict):
    """Creates a new instance of a model with its given configuration."""
    model_name = configuration["base_model"]
    base_class = getattr(
        importlib.import_module("models.mask_models"), model_name
    )
    model = base_class(configuration)
    return model
