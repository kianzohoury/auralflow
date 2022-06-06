# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from .losses import (
    WeightedComponentLoss,
    component_loss,
    KLDivergenceLoss,
    kl_div_loss,
    L1Loss,
    L2Loss,
    L2MaskLoss,
    RMSELoss,
    SIDRLoss,
    get_evaluation_metrics,
)
from typing import Union, Callable


import torch.nn as nn


__all__ = [
    "WeightedComponentLoss",
    "component_loss",
    "KLDivergenceLoss",
    "kl_div_loss",
    "L1Loss",
    "L2Loss",
    "L2MaskLoss",
    "RMSELoss",
    "SIDRLoss",
    "get_model_criterion",
    "get_evaluation_metrics",
    "RMSELoss"
]


def get_model_criterion(model, config: dict) -> Union[nn.Module, Callable]:
    """Gets model criterion according to configuration file."""
    loss_fn = config["training_params"]["criterion"]
    model_type = config["model_params"]["model_type"]
    is_vae_model = model_type == "SpectrogramNetVAE"
    if loss_fn == "component_loss":
        criterion = WeightedComponentLoss(
            model=model,
            alpha=config["training_params"]["alpha_constant"],
            beta=config["training_params"]["beta_constant"],
        )
    elif is_vae_model and loss_fn == "kl_div_loss":
        criterion = KLDivergenceLoss(model=model, loss_fn=loss_fn)
    elif loss_fn == "l1":
        criterion = L1Loss(model=model)
    elif loss_fn == "rmse":
        criterion = RMSELoss(model=model)
    elif loss_fn == "l2_mask_loss":
        criterion = L2MaskLoss(model=model)
    elif loss_fn == "si_sdr_loss":
        criterion = SIDRLoss(model=model)
    else:
        criterion = L2Loss(model=model)
    return criterion
