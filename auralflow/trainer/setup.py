# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git
import functools
import inspect
import json

import torch.nn as nn
from prettytable import PrettyTable

from auralflow.losses import *
from auralflow.models import SeparationModel, SpectrogramMaskModel, SPEC_MODELS, AUDIO_MODELS, ALL_MODELS
from auralflow.utils import save_config, load_config
from dataclasses import asdict, dataclass, Field, fields, MISSING
from typing import List, Union, Optional, Tuple, Dict


@dataclass(frozen=True)
class Config:
    """Base class for all configurations."""

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, **kwargs) -> 'Config':
        """Filters out keyword arguments before creating a new instance."""
        valid_args = [field.name for field in fields(cls)]
        filtered_args = {
            key: val for (key, val) in kwargs.items() if key in valid_args
        }
        return cls(**filtered_args)

    @classmethod
    def defaults(cls) -> List[Field]:
        """Returns the default training configuration keyword arguments."""
        default_fields = []
        for field in fields(cls):
            if field.default is not MISSING:
                print(field.type.__origin__, field.type.__args__)

                default_fields.append(field)
        return default_fields

    def save(self, filepath: str) -> None:
        """Saves the configuration given a filepath."""
        save_config(config=asdict(self), save_filepath=filepath)


@dataclass(frozen=True)
class AudioModelConfig(Config):
    """Specifies the build configuration for audio-based models."""

    model_type: str
    targets: List[str]
    num_channels: int = 1
    num_hidden_channels: int = 16
    sample_length: int = 3
    sample_rate: int = 44100
    dropout_p: float = 0.4
    leak_factor: float = 0
    normalize_input: bool = True
    normalize_output: bool = True

    # Additional default parameters for LSTM models.
    recurrent_depth: int = 3
    hidden_size: int = 1024
    input_axis: int = 1

    def __str__(self) -> str:
        """Displays the model configuration as a table."""
        param_table = PrettyTable(
            field_names=["Parameter", "Value"],
            align="l",
            title="Model Configuration",
            min_width=21
        )
        for param_label, param in asdict(self).items():
            param_table.add_row([param_label, param])
        return str(param_table)


@dataclass(frozen=True)
class SpecModelConfig(AudioModelConfig):
    """Specifies the build configuration for spectrogram-based models."""

    mask_act_fn: str = "sigmoid"
    num_fft: int = 1024
    window_size: int = 1024
    hop_length: int = 512


@dataclass(frozen=True)
class CriterionConfig(Config):
    """Specifies the model loss criterion for all model types."""

    input_type: str
    criterion: str = "si_sdr"
    construction_loss: str = "l2"
    reduction: str = "mean"
    best_perm: bool = True
    alpha: float = 0.2
    beta: float = 0.8


@dataclass(frozen=True)
class VisualsConfig(Config):
    """Specifies all visualization options for tensorboard logging."""

    logging_dir: Optional[str]
    tensorboard: bool = True
    view_norm: bool = True
    view_epoch: bool = True
    view_iter: bool = True
    view_grad: bool = True
    view_weights: bool = True
    view_spec: bool = True
    view_wave: bool = True
    play_estimate: bool = True
    play_residual: bool = True
    image_dir: Optional[str] = None
    image_freq: int = 5
    silent: bool = False


@dataclass(frozen=True)
class TrainingConfig(Config):
    """Specifies all parameters and settings for running model training."""

    criterion_config: CriterionConfig
    visuals_config: VisualsConfig
    checkpoint: str
    device: str
    use_amp: bool = True
    scale_grad: bool = True
    clip_grad: bool = True
    lr: float = 0.008
    lr_lstm: float = lr * 1e-3
    init_scale: float = 2.0 ** 16
    max_grad_norm: Optional[float] = 100.0
    max_plateaus: int = 5
    stop_patience: int = 5
    min_delta: float = 0.01
    max_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    pre_fetch: int = 4


def _create_model_config(
    model_type: str, targets: List[str], **kwargs
) -> AudioModelConfig:
    data_class = SpecModelConfig if model_type in SPEC_MODELS else AudioModelConfig
    constructor_params = inspect.signature(data_class.__init__).parameters
    filtered_args = {}
    for key, val in kwargs.items():
        if key in constructor_params:
            filtered_args[key] = val
    return data_class(model_type=model_type, targets=targets, **filtered_args)


def _load_model_config(filepath: str) -> AudioModelConfig:
    """Loads configuration data from a file to create a new instance."""
    try:
        with open(filepath) as config_file:
            config_data = json.load(fp=config_file)
            model_type = config_data["model_type"]
            cls = SpecModelConfig if model_type in SPEC_MODELS else AudioModelConfig
            return cls(**config_data)
    except IOError as error:
        raise error


def _build_from_config(
    model_config: AudioModelConfig, device: str = 'cpu'
) -> SeparationModel:
    r"""Creates a new ``SeparationModel`` instance given custom specifications.

    Note that the model's parameters will be in some initial state (instead of
    some previous training state/checkpoint).

    Args:
        model_config (AudioModelConfig): Model configuration settings.
        device (str): Device to load model onto. Default: ``cpu``.
    Returns:
        SeparationModel: Customized model.
    """
    if isinstance(model_config, SpecModelConfig):
        model = SpectrogramMaskModel(
            model_type=model_config.model_type,
            targets=model_config.targets,
            num_fft=model_config.num_fft,
            window_size=model_config.window_size,
            hop_length=model_config.hop_length,
            sample_length=model_config.sample_length,
            sample_rate=model_config.sample_rate,
            num_channels=model_config.num_channels,
            num_hidden_channels=model_config.num_hidden_channels,
            mask_act_fn=model_config.mask_act_fn,
            leak_factor=model_config.leak_factor,
            dropout_p=model_config.dropout_p,
            normalize_input=model_config.normalize_input,
            normalize_output=model_config.normalize_output,
            recurrent_depth=model_config.recurrent_depth,
            hidden_size=model_config.hidden_size,
            input_axis=model_config.input_axis,
            device=device
        )
    else:
        # TODO: Audio-based models.
        model = None
    return model


def _get_loss_criterion(criterion_config: CriterionConfig) -> nn.Module:
    """Returns a loss criterion given its specific configuration."""
    if criterion_config.input_type == "spectrogram":
        if criterion_config.criterion == "component":
            criterion = ComponentLoss(
                alpha=criterion_config.alpha, beta=criterion_config.beta
            )
        elif criterion_config.criterion == "kl_div":
            criterion = KLDivergenceLoss(
                loss_fn=criterion_config.construction_loss
            )
        elif criterion_config.criterion == "l2":
            criterion = L2Loss(
                reduce_mean=criterion_config.reduction == "mean"
            )
        elif criterion_config.criterion == "l1":
            criterion = L1Loss(
                reduce_mean=criterion_config.reduction == "mean"
            )
        elif criterion_config.criterion == "mask":
            criterion = MaskLoss(
                loss_fn=criterion_config.construction_loss,
                reduce_mean=criterion_config.reduction == "mean"
            )
        elif criterion_config.criterion == "si_sdr":
            criterion = SISDRLoss(best_perm=criterion_config.best_perm)
        elif criterion_config.criterion == "rmse":
            # TODO: RMSE loss.
            criterion = None
        else:
            criterion = None
    else:
        # TODO: Audio-based models.
        criterion = None
    return criterion
