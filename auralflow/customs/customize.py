# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from auralflow.losses import *
from auralflow.models import *
from pathlib import Path
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Callable


def init_model(configuration: dict) -> SeparationModel:
    """Creates a new model instance given configuration data.

    Args:
        configuration (dict): Model configuration data.
    """
    model_type = configuration["model_params"]["model_type"]
    if model_type in model_names:
        # Only allow SpectrogramMaskModel for now.
        model = SpectrogramMaskModel(configuration)
    else:
        model = None
    return model


def get_model_criterion(
    model: SeparationModel, config: dict
) -> Union[nn.Module, Callable]:
    """Gets model criterion according to its configuration file."""
    loss_fn = config["training_params"]["criterion"]
    model_type = config["model_params"]["model_type"]
    is_vae_model = model_type == "SpectrogramNetVAE"
    if loss_fn == "component_loss":
        criterion = ComponentLoss(
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
        criterion = SISDRLoss(model=model)
    else:
        criterion = L2Loss(model=model)
    return criterion


def setup_model(model: SeparationModel) -> SeparationModel:
    """Sets up a separation model according to its internal configuration.

    If model is in training mode, it creates and loads objects related to
    training; otherwise, it loads the best version of the model.


    Args:
        model (SeparationModel): Separation model.

    Returns:
        SeparationModel: Separation model.

    Raises:
        OSError: Raised if the model cannot be loaded.
        FileNotFoundError: Raised if a checkpoint file cannot be found.
    """
    if model.training_mode:
        last_epoch = model.training_params["last_epoch"]

        # Define model criterion.
        model.criterion = get_model_criterion(model, config=model.config)
        model.train_losses, model.val_losses = [], []

        # Define optimizer.
        if isinstance(model.model, SpectrogramNetLSTM):
            param1, param2 = model.model.split_lstm_parameters()
            params = [
                {"params": param1, "lr": model.training_params["lr"] * 1e-4},
                {"params": param2, "lr": model.training_params["lr"]},
            ]
        else:
            param1 = model.model.parameters()
            params = [{"params": param1}]
        model.optimizer = Adam(params=params, lr=model.training_params["lr"])

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
            growth_interval=20000,
        )

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
    else:
        try:
            best_epoch = model.training_params["best_epoch"]
            model.load_model(global_step=best_epoch)
            model.training_mode = False
        except (OSError, FileNotFoundError) as error:
            print(f"Failed to load model {model.model_name}.")
            raise error
    return model
