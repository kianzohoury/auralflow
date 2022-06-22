# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch


from auralflow.losses import *
from auralflow.models import *
from auralflow.utils import load_object
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Only allow SpectrogramMaskModel class for now.
ALL_MODEL_NAMES = {
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE"
}
SPEC_MASK_MODELS = set(ALL_MODEL_NAMES)


#
# # Set other attributes.
# model._model_name = self.model_params["model_name"]
#
# model._checkpoint_path = self.model_params["save_dir"] + "/checkpoint"
# model._silent_checkpoint = self.training_params["silent_checkpoint"]
# model._training_mode = self.training_params["training_mode"]
#

def init_model(configuration: dict) -> SeparationModel:
    r"""Creates a new ``SeparationModel`` instance given a configuration.

    Reads the given configuration data and builds a new ``SeparationModel``
    according to the spec, if possible. See ``auralflow.utils.load_config`` for
    loading configuration files as data.

    Args:
        configuration (dict): Model configuration data.

    Returns:
        SeparationModel: New ``SeparationModel`` instance.

    Raises:
        ValueError: Raised if the requested base model does not exist.
        KeyError: Raised if a parameter is not specified or included in the
            configuration data.

    Examples:

        Imports:

            >>> import auralflow.utils
            >>> import os

        Create model folder with a new configuration file:

            >>> model_dir = os.getcwd() + "/my_model"
            >>> auralflow.utils.copy_config_template(save_dir=model_dir)
            >>> config_data = auralflow.utils.load_config(save_dir=model_dir)
            >>> config_data.keys() # doctest: +NORMALIZE_WHITESPACE
            dict_keys(['model_params', 'dataset_params', 'training_params',
            'visualizer_params'])

        Instantiate a new model with the above configuration data:

            >>> separation_model = init_model(configuration=config_data)
            >>> type(separation_model)
            <class 'models.SeparationModel'>
    """
    try:
        model_params = configuration["model_params"]
        dataset_params = configuration["dataset_params"]
        base_model_type = model_params["model_type"]
        if base_model_type in ALL_MODEL_NAMES:
            if base_model_type in SPEC_MASK_MODELS:
                model = SpectrogramMaskModel(
                    base_model_type=base_model_type,
                    target_labels=model_params["targets"],
                    num_fft=dataset_params["num_fft"],
                    window_size=dataset_params["window_size"],
                    hop_length=dataset_params["hop_length"],
                    sample_length=dataset_params["sample_length"],
                    sample_rate=dataset_params["sample_rate"],
                    num_channels=dataset_params["num_channels"],
                    num_hidden_channels=model_params["hidden_channels"],
                    mask_act_fn=model_params["mask_activation"],
                    leak_factor=model_params["leak_factor"],
                    dropout_p=model_params["dropout_p"],
                    normalize_input=model_params["normalize_input"],
                    normalize_output=model_params["normalize_output"],
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                # TODO: Audio-based models.
                model = None
        else:
            raise ValueError(f"{base_model_type} is not a valid base model.")
    except KeyError as error:
        raise error
    return model


def set_model_criterion(model: SeparationModel):
    """Sets the model criterion according to its configuration file."""
    loss_fn = model.training_params["criterion"]
    model_type = model.model_params["model_type"]
    is_vae_model = model_type == "SpectrogramNetVAE"
    if loss_fn == "component_loss":
        criterion = ComponentLoss(
            model=model,
            alpha=model.training_params["alpha"],
            beta=model.training_params["beta"],
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
    # Set the criterion.
    model.criterion = criterion


def setup_model(model: SeparationModel) -> SeparationModel:
    r"""Sets up a ``SeparationModel`` according to its internal configuration.

    If the model is in training mode, it loads the states of (or creates)
    various model attributes, which include:
        - the model's underlying network (as a ``nn.Module``)
        - the optimizer
        - the learning rate scheduler
        - the gradient scaler (if applicable)
        - automatic mixed precision (if applicable)
        - misc. model attributes

    Otherwise, it loads the best version of the trained model.

    Args:
        model (SeparationModel): Separation model.

    Returns:
        SeparationModel: Separation model.

    Raises:
        OSError: Raised if the model cannot be loaded.
        FileNotFoundError: Raised if a checkpoint file cannot be found.

    Examples:
        >>> import auralflow.utils as utils
        >>> import os
        >>>
        >>> # folder directory
        >>> model_dir = os.getcwd() + "/my_model"
        >>>
        >>> # write a new configuration file
        >>> utils.copy_config_template(save_dir=model_dir)
        >>>
        >>> # load the configuration file as a dictionary
        >>> config_data = utils.load_config(save_dir=model_dir)
        >>>
        >>> # instantiate the model
        >>> my_model = init_model(configuration=config_data)
        >>>
        >>> # set up the model for training
        >>> my_model = setup_model(my_model)
        >>>
        >>> # check that some model attributes have been filled
        >>> hasattr(my_model, "optimizer")
        True
        >>> hasattr(my_model, "criterion")
        True
        >>> hasattr(my_model, "scheduler")
        True
    """
    if model._training_mode:
        last_epoch = model.training_params["last_epoch"]

        # Define model criterion.
        set_model_criterion(model)
        model.train_losses, model.val_losses = [], []

        # Define optimizer.
        if isinstance(model.model, SpectrogramNetLSTM):
            param1, param2 = model.model.split_lstm_parameters()

            # LSTM layers will start with a much smaller learning rate.
            params = [
                {"params": param1, "lr": model.training_params["lr"] * 1e-4},
                {"params": param2, "lr": model.training_params["lr"]},
            ]
        else:
            param1 = model.model.parameters()
            params = [{"params": param1}]
        model.optimizer = Adam(params=params, lr=model.training_params["lr"])

        # Define lr scheduler and early stopping params.
        model._max_lr_steps = model.training_params["max_lr_steps"]
        model._stop_patience = model.training_params["stop_patience"]
        model.scheduler = ReduceLROnPlateau(
            optimizer=model.optimizer,
            mode="min",
            verbose=True,
            patience=model._stop_patience,
        )

        # Initialize gradient scaler. Will only be invoked if using AMP.
        use_amp = model.training_params["use_mixed_precision"]
        model._use_amp = use_amp and model.device == "cuda"
        model._grad_scaler = GradScaler(
            init_scale=model.training_params["mixed_precision_scale"],
            enabled=model._use_amp,
            growth_factor=100,
            growth_interval=20000,
        )

        if model.training_params["last_epoch"] >= 0:
            # Load model, optim, scheduler and scaler states.
            load_object(model=model, obj_name="model", global_step=last_epoch)
            load_object(
                model=model, obj_name="optimizer", global_step=last_epoch
            )
            load_object(
                model=model, obj_name="scheduler", global_step=last_epoch
            )
            if model.training_params["use_mixed_precision"]:
                load_object(
                    model=model, obj_name="grad_scaler", global_step=last_epoch
                )
        else:
            # Create checkpoint folder.
            Path(model._checkpoint_path).mkdir(exist_ok=True)
    else:
        try:
            best_epoch = model.training_params["best_epoch"]
            load_object(model=model, obj_name="model", global_step=best_epoch)
            model._training_mode = False
        except (OSError, FileNotFoundError) as error:
            print(f"Failed to load model {model._model_name}.")
            raise error
    return model


