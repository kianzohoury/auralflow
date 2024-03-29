# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

"""Source separation model wrappers and ``torch.nn.Module`` models."""

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
