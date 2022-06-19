# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from .architectures import *
from .base import SeparationModel
from .mask_model import SpectrogramMaskModel


__all__ = [
    "SeparationModel",
    "SpectrogramMaskModel",
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE",
    "model_names",
]

model_names = __all__[2:]

__doc__ = r"""Separation model base classes, implementations and underlying
PyTorch ``nn.Module`` classes."""