# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import json
import torch.nn as nn


from auralflow import losses
from auralflow import models
from dataclasses import asdict, dataclass, fields, Field, MISSING
from prettytable import PrettyTable
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Config:
    """Base class for data configurations specified through CLI input."""

    def __init__(self, *args, **kwargs) -> None:
        super(Config, self).__init__(*args, **kwargs)

    @classmethod
    def defaults(cls) -> List[Field]:
        """Returns the default training configuration keyword arguments."""
        return _get_default_fields(cls)

    @classmethod
    def from_dict(cls, **kwargs) -> 'Config':
        """Returns a new config instance from keyword arguments."""
        valid_args = [field.name for field in fields(cls)]
        filtered_args = {
            key: val for (key, val) in kwargs.items() if key in valid_args
        }
        return cls(**filtered_args)

    def save(self, filepath: str) -> None:
        """Saves the configuration to the given filepath."""
        with open(filepath, "w") as config_file:
            json.dump(obj=asdict(self), fp=config_file, indent=4)


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
    normalize_input: bool = False
    normalize_output: bool = False

    # Additional default parameters for LSTM models.
    recurrent_depth: int = 3
    hidden_size: int = 1024
    input_axis: int = 1

    def __str__(self) -> str:
        if "LSTM" not in self.model_type:
            excluded_fields = ["recurrent_depth", "hidden_size", "input_axis"]
        else:
            excluded_fields = []
        return _config_to_str(
            config=self, title="model config", excluded_fields=excluded_fields
        )


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

    def __str__(self) -> str:
        if criterion != "component":
            excluded_fields = ["alpha", "beta"]
        if criterion == "kl_div":
            excluded_fields.append("construction_loss")
        return _config_to_str(
            config=self, title="model config", excluded_fields=excluded_fields
        )


@dataclass(frozen=True)
class VisualsConfig(Config):
    """Specifies all visualization options for tensorboard logging."""

    image_dir: str
    logging_dir: str
    tensorboard: bool = True
    view_norm: bool = True
    view_epoch: bool = True
    view_iter: bool = True
    view_grad: bool = False
    view_weights: bool = False
    view_spec: bool = False
    view_wave: bool = False
    play_estimate: bool = False
    play_residual: bool = False
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
    max_grad_norm: float = 100.0
    max_plateaus: int = 5
    stop_patience: int = 5
    min_delta: float = 0.01
    max_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    pre_fetch: int = 4

    @classmethod
    def defaults(cls) -> List[Field]:
        """Returns the default training arguments."""
        field_defaults = super().defaults()
        field_defaults += CriterionConfig.defaults()
        field_defaults += VisualsConfig.defaults()
        return field_defaults

    def __str__(self):
        return _config_to_str(config=self, title="training config")


def _get_default_fields(cls) -> List[Field]:
    """Helper function that returns the field defaults for the given class."""
    default_fields = []
    for field in fields(cls):
        if field.default is not MISSING:
            # Handle optional-type fields.
            if hasattr(field.type, "__args__"):
                union_types = field.type.__args__
                if len(union_types) == 2 and union_types[-1] is type(None):
                    field.type = union_types[0]
                else:
                    continue
            default_fields.append(field)
    return default_fields


def _create_model_config(
    model_type: str, targets: List[str], **kwargs
) -> Config:
    """Returns a model configuration corresponding to the provided specs."""
    cls = SpecModelConfig if "spec" in model_type.lower() else AudioModelConfig
    return cls.from_dict(model_type=model_type, targets=targets, **kwargs)


def _load_model_config(filepath: str) -> Config:
    """Returns a model configuration read from a previous config file."""
    try:
        with open(filepath, "r") as config_file:
            config_data = json.load(fp=config_file)
            return _create_model_config(**config_data)
    except IOError as error:
        raise error


def _build_model(
    model_config: AudioModelConfig, device: str = 'cpu'
) -> models.SeparationModel:
    r"""Initializes a ``SeparationModel`` given a model configuration.

    Note that the model's parameters will be in some initial state (instead of
    some previous training state/checkpoint).
    """
    if isinstance(model_config, SpecModelConfig):
        model = models.SpectrogramMaskModel(
            **model_config.__dict__, device=device
        )
    else:
        # TODO: Audio-based models.
        raise NotImplementedError
    return model


def _get_loss_criterion(criterion_config: CriterionConfig) -> nn.Module:
    """Returns the model loss criterion according to its configuration data."""
    if criterion_config.input_type == "spectrogram":
        if criterion_config.criterion == "component":
            criterion = losses.ComponentLoss(
                alpha=criterion_config.alpha, beta=criterion_config.beta
            )
        elif criterion_config.criterion == "kl_div":
            criterion = losses.KLDivergenceLoss(
                loss_fn=criterion_config.construction_loss
            )
        elif criterion_config.criterion == "l2":
            criterion = losses.L2Loss(
                reduce_mean=criterion_config.reduction == "mean"
            )
        elif criterion_config.criterion == "l1":
            criterion = losses.L1Loss(
                reduce_mean=criterion_config.reduction == "mean"
            )
        elif criterion_config.criterion == "mask":
            criterion = losses.MaskLoss(
                loss_fn=criterion_config.construction_loss,
                reduce_mean=criterion_config.reduction == "mean"
            )
        elif criterion_config.criterion == "si_sdr":
            criterion = losses.SISDRLoss(best_perm=criterion_config.best_perm)
        elif criterion_config.criterion == "rmse":
            # TODO: RMSE loss.
            raise NotImplementedError
        else:
            raise ValueError(
                f"{criterion_config.criterion} is not a valid loss criterion."
            )
    else:
        # TODO: Audio-based models.
        raise NotImplementedError
    return criterion


def _config_to_str(
    config: Config, title: str, excluded_fields: Optional[List[str]] = None
) -> str:
    """Returns a string representation for the given configuration."""
    param_table = PrettyTable(
        field_names=["parameter", "value"],
        align="l",
        title=title,
        min_width=21
    )
    excluded_fields = excluded_fields if excluded_fields is not None else []
    for param_label, param in _flatten_dict(asdict(config)).items():
        if param_label not in excluded_fields:
            param_table.add_row([param_label, param])
    return str(param_table)


def _flatten_dict(d: Dict) -> Dict:
    flattened = {}
    for key, val in d.items():
        if isinstance(val, dict):
            flattened.update(_flatten_dict(val))
        else:
            flattened[key] = val
    return flattened
