import importlib

mask_models = ["SpectrogramMaskModel"]


def create_model(configuration: dict):
    model_name = configuration["base_model"]
    base_class = getattr(importlib.import_module("models.mask_models"), model_name)
    model = base_class(configuration)
    return model