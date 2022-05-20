# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch


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
from utils import save_config


__all__ = [
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE",
    "SpectrogramMaskModel",
    "SeparationModel",
    "create_model",
    "setup_model",
    "save_object",
    "load_object"
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



# def load_pretrained_model(checkpoint_path: str):
#     try:
#         model = torch.load(f=checkpoint_path)


def _add_checkpoint_tag(filename: str, obj_name: str, global_step: int) -> str:
    """Attaches a suffix to the checkpoint filename depending on the object."""
    if obj_name == "model":
        filename += f"_{global_step}.pth"
    elif obj_name == "optimizer":
        filename += f"_optimizer_{global_step}.pth"
    elif obj_name == "scheduler":
        filename += f"_scheduler_{global_step}.pth"
    elif obj_name == "grad_scaler":
        filename += f"_autocast_{global_step}.pth"
    return filename


def save_object(model_wrapper, obj_name: str, global_step: int) -> None:
    """Saves object state as .pth file under the checkpoint directory."""
    filename = f"{model_wrapper.checkpoint_path}/{model_wrapper.model_name}"

    # Get object-specific filename.
    filename = _add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )
    if hasattr(model_wrapper, obj_name):
        # Retrieve object's state.
        if obj_name == "model":
            state_dict = getattr(model_wrapper, obj_name).cpu().state_dict()
            # Transfer model back to GPU if applicable.
            model_wrapper.model.to(model_wrapper.device)
        else:
            state_dict = getattr(model_wrapper, obj_name).state_dict()
        try:
            # Save object's state to filename.
            torch.save(state_dict, f=filename)
            if not model_wrapper.training_params["silent_checkpoint"]:
                print(f"Successfully saved {obj_name}.")
        except OSError as error:
            print(f"Failed to save {obj_name} state.")
            raise error


def load_object(model_wrapper, obj_name: str, global_step: int) -> None:
    """Loads object and attaches it to model_wrapper and its device."""
    filename = f"{model_wrapper.checkpoint_path}/{model_wrapper.model_name}"

    # Get object-specific filename.
    filename = _add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )
    try:
        # Try to read object state from the given file.
        state_dict = torch.load(filename, map_location=model_wrapper.device)
    except (OSError, FileNotFoundError) as error:
        print(f"Failed to load {obj_name} state.")
        raise error
    if hasattr(model_wrapper, obj_name):
        # Load state into object.
        getattr(model_wrapper, obj_name).load_state_dict(state_dict)
        print(f"Loaded {obj_name} successfully.")


def setup_model(model: SeparationModel) -> None:
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
            # Create checkpoint folder, save copy of config file to it.
            Path(model.checkpoint_path).mkdir(exist_ok=True)
            save_config(model.config, model.checkpoint_path)

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
            use_amp = model.training_params["use_mixed_precision"]
            model.use_amp = use_amp and model.device == "cuda"
            model.grad_scaler = GradScaler(
                init_scale=model.training_params["mixed_precision_scale"],
                enabled=model.use_amp,
                growth_factor=100,
                growth_interval=20000
            )
    else:
        pass




