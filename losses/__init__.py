from . import losses
from typing import Union, Callable

import torch.nn as nn


def get_model_criterion(model, config: dict) -> Union[nn.Module, Callable]:
    """Gets model criterion according to configuration file."""
    loss_fn = config["training_params"]["criterion"]
    model_type = config["model_params"]["model_type"]
    is_vae_model = model_type == "SpectrogramLSTMVariational"
    if loss_fn == "component_loss":
        criterion = losses.WeightedComponentLoss(
            model=model,
            alpha=config["training_params"]["alpha_constant"],
            beta=config["training_params"]["beta_constant"]
        )
    elif is_vae_model:
        criterion = losses.KLDivergenceLoss(
            model=model,
            loss_fn=loss_fn
        )
    elif loss_fn == "l1":
        criterion = losses.L1Loss(model=model)
    else:
        criterion = losses.L2Loss(model=model)
    return criterion
