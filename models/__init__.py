# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from .architectures import (
    SpectrogramNetSimple,
    SpectrogramNetLSTM,
    SpectrogramNetVAE,
)
from .base import SeparationModel
from .mask_model import SpectrogramMaskModel
from pathlib import Path


__all__ = [
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE",
    "SeparationModel",
    "SpectrogramMaskModel",
    "create_model",
    "setup_model",
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


def setup_model(model: SeparationModel):
    if model.training_mode:
        last_epoch = model.training_params["last_epoch"]
        if model.training_params["last_epoch"] >= 0:
            model.load_model(global_step=last_epoch)
            model.load_optim(global_step=last_epoch)
            model.load_scheduler(global_step=last_epoch)
            if model.training_params["use_mixed_precision"]:
                model.load_grad_scaler(global_step=last_epoch)
        else:
            Path(model.checkpoint_path).mkdir(exist_ok=True)
            pass

        # load model, load scheduler, load optimizer, load scaler,

    else:
        pass
