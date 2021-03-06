# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

"""Separation models and PyTorch ``nn.Module`` classes."""

__all__ = [
    "SeparationModel",
    "SpectrogramMaskModel",
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE",
]

from .architectures import (
    SpectrogramNetSimple,
    SpectrogramNetLSTM,
    SpectrogramNetVAE
)
from .base import SeparationModel
from .mask_model import SpectrogramMaskModel

# Only spectrogram-based models for now.
SPEC_MODELS = {
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE"
}
AUDIO_MODELS = set()
ALL_MODELS = SPEC_MODELS | AUDIO_MODELS
