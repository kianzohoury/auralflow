# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

__all__ = [
    "ModelTrainer",
    "CallbackManager",
    "TrainingCallback",
    "AudioPlayerCallback",
    "LayersVisualCallback",
    "LossCallback",
    # "MetricsCallback",
    "SpectrogramVisualCallback",
    "WaveformVisualCallback",
]

from .callbacks import (
    AudioPlayerCallback,
    CallbackManager,
    _create_callbacks,
    LayersVisualCallback,
    LossCallback,
    # MetricsCallback,
    SpectrogramVisualCallback,
    TrainingCallback,
    WaveformVisualCallback,
)
from .trainer import (
    ModelTrainer,
    _DefaultModelTrainer
)

from .setup import (
    _build_from_config,
    _parse_to_config,
    _get_loss_criterion,
    CriterionConfig,
    AudioModelConfig,
    SpecModelConfig,
    TrainingConfig,
)
