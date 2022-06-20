# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from .customs import init_model, setup_model


__all__ = [
    "init_model",
    "setup_model",
]

__doc__ = """
Model instantiation and setup for training, evaluation and testing.
"""