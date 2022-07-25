# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

"""
Functional losses and ``nn.Module`` losses for integration with the
``SeparationModel`` class.
"""

__all__ = [
    "component_loss",
    "kl_div_loss",
    "si_sdr_loss",
    # "rmse_loss",
    "get_evaluation_metrics",
    "ComponentLoss",
    "KLDivergenceLoss",
    "SISDRLoss",
    "L1Loss",
    "L2Loss",
    # "RMSELoss",
    "MaskLoss"
]

from .losses import (
    component_loss,
    kl_div_loss,
    si_sdr_loss,
    # rmse_loss,
    get_evaluation_metrics,
    ComponentLoss,
    KLDivergenceLoss,
    SISDRLoss,
    L1Loss,
    L2Loss,
    # RMSELoss,
    MaskLoss,
)


CRITERION_NAMES = [
    "kl_div",
    "l1",
    "l2",
    "mask",
    "si_sdr",
    "rmse"
]

CONSTRUCTION_LOSS_NAMES = ["l1", "l2"]