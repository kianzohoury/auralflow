# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git
from torch.cuda.amp import GradScaler

from .architectures import (
    SpectrogramNetSimple,
    SpectrogramNetLSTM,
    SpectrogramNetVAE,
)
from .base import SeparationModel
from .mask_model import SpectrogramMaskModel
from losses import get_model_criterion
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
            # Load model, optim, scheduler and scaler states.
            model.load_model(global_step=last_epoch)
            model.load_optim(global_step=last_epoch)
            model.load_scheduler(global_step=last_epoch)
            if model.training_params["use_mixed_precision"]:
                model.load_grad_scaler(global_step=last_epoch)
        else:
            # Create checkpoint folder.
            Path(model.checkpoint_path).mkdir(exist_ok=True)

            # Define model criterion.
            model.criterion = get_model_criterion(model, config=model.config)
            model.train_losses, model.val_losses = [], []

            # Define optimizer.
            model.optimizer = AdamW(
                params=model.model.parameters(), lr=model.training_params["lr"]
            )

            # Define lr scheduler and early stopping params.
            model.max_lr_steps = model.training_params["max_lr_steps"]
            model.stop_patience = model.training_params["stop_patience"]
            model.scheduler = ReduceLROnPlateau(
                optimizer=model.optimizer,
                mode="min",
                verbose=True,
                patience=model.stop_patience,
            )

            # Initialize gradient scaler. Will only be invoked if using AMP.
            enable_amp = model.training_params["use_mixed_precision"]
            model.grad_scaler = GradScaler(
                init_scale=model.training_params["mixed_precision_scale"],
                enabled=enable_amp and model.device == "cuda",
            )
    else:
        pass




