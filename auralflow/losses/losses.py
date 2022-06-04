# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from collections import OrderedDict
from typing import Any, Dict, Mapping

import asteroid.metrics
import torch
import torch.nn as nn
from torch import FloatTensor, Tensor
from torch.nn import functional
from auralflow.utils.data_utils import trim_audio
from torchaudio.transforms import Resample


def component_loss(
    mask: FloatTensor,
    target: FloatTensor,
    residual: FloatTensor,
    alpha: float = 0.2,
    beta: float = 0.8,
) -> Tensor:
    """Weighted L2 loss using 2 or 3 components depending on arguments.

    Balances the target source separation quality versus the amount of
    residual noise attenuation. Optional third component balances the
    quality of the residual noise against the other two terms.
    """

    filtered_target = mask * target
    filtered_residual = mask * residual

    # Separation quality term.
    total_separation_loss = torch.sum((filtered_target - target) ** 2)

    # Noise attenuation term.
    total_noise_loss = torch.sum(filtered_residual**2)

    summed_dims = list(range(1, filtered_residual.dim()))
    filtered_res_unit = filtered_residual / torch.linalg.norm(
        filtered_residual, dim=summed_dims, keepdim=True
    )
    target_res_unit = residual / torch.linalg.norm(
        residual, dim=summed_dims, keepdim=True
    )

    # Noise quality term.
    total_noise_quality_loss = torch.sum(
        (filtered_res_unit - target_res_unit) ** 2
    )

    # Constrain alpha + beta <= 1.
    if alpha + beta > 1:
        total = alpha + beta + 1e-8
        alpha = alpha / total
        beta = beta / total

    # Combine loss components.
    loss = (1 - alpha - beta) * total_separation_loss
    loss = loss + alpha * total_noise_loss
    loss = loss + beta * total_noise_quality_loss
    mean_loss = loss / filtered_target.numel()
    return mean_loss


def l1_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """L1 loss."""
    return functional.l1_loss(estimate, target)


def l2_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """L2 loss."""
    return functional.mse_loss(estimate, target)


def kl_div_loss(mu: FloatTensor, sigma: FloatTensor) -> Tensor:
    """Computes KL term using the closed form expression.

    KL term is defined as := D_KL(P||Q), where P is the modeled distribution,
    and Q is a standard normal N(0, 1). The term is combined with the
    reconstruction loss.
    """
    return 0.5 * torch.mean(mu**2 + sigma**2 - torch.log(sigma**2) - 1)


def get_evaluation_metrics(
    mixture: Tensor,
    estimate: Tensor,
    target: Tensor,
    sr: int = 16000,
    num_batch: int = 8,
) -> Mapping[str, float]:
    """Returns batch-wise means of standard source separation eval scores."""

    # Unsqueeze batch dimension if audio is unbatched.
    mixture = mixture.unsqueeze(0) if mixture.dim() == 2 else mixture
    estimate = estimate.unsqueeze(0) if estimate.dim() == 2 else estimate
    target = target.unsqueeze(0) if target.dim() == 2 else target

    # Reduce sample rate.
    resampler = Resample(orig_freq=44100, new_freq=sr)
    mixture = resampler(mixture.cpu())
    estimate = resampler(estimate.cpu())
    target = resampler(target.cpu())

    # Collapse channels to mono, convert to numpy arrays, trim audio clips.
    mixture = torch.mean(mixture, dim=1, keepdim=True).squeeze(-1).numpy()
    estimate = torch.mean(estimate, dim=1, keepdim=True).squeeze(-1).numpy()
    target = torch.mean(target, dim=1, keepdim=True).squeeze(-1).numpy()
    mixture, estimate, target = trim_audio([mixture, estimate, target])

    running_metrics = {
        "pesq": 0,
        "sar": 0,
        "sdr": 0,
        "si_sdr": 0,
        "sir": 0,
        "stoi": 0,
    }

    num_batch = min(num_batch, mixture.shape[0])

    for i in range(num_batch):
        # Compute metrics.
        named_metrics = asteroid.metrics.get_metrics(
            mix=mixture[i],
            clean=target[i],
            estimate=estimate[i],
            sample_rate=sr,
            metrics_list="all",
            ignore_metrics_errors=True,
            average=True,
        )
        # Accumulate metrics.
        for metric_name, val in named_metrics.items():
            if val is not None and metric_name in running_metrics:
                running_metrics[metric_name] += val

    metrics = {}
    # Compute batch-wise average metrics.
    for metric_name, val in running_metrics.items():
        mean_val = val / mixture.shape[0]
        metrics[metric_name] = mean_val
    return metrics


class WeightedComponentLoss(nn.Module):
    """Wrapper class for calling weighted component loss."""

    def __init__(
        self, model, alpha: float, beta: float, regularizer: bool = True
    ) -> None:
        super(WeightedComponentLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer
        # if regularizer:
        #     self.reg_loss = l2_loss

    def forward(self):
        """Calculates a weighted component loss."""

        # Compute weighted loss.
        self.model.batch_loss = component_loss(
            mask=self.model.mask,
            target=self.model.target,
            residual=self.model.mixture - self.model.target,
            alpha=self.alpha,
            beta=self.beta,
        )

        # Add kl term if using VAE.
        if hasattr(self.model, "get_kl_div"):
            kl_term = self.model.get_kl_div()
            self.model.batch_loss = self.model.batch_loss + kl_term

        # if self.regularizer:
        #     self.model.batch_loss = self.model.batch_loss + self.reg_loss(
        #         self.model.estimate, self.model.target
        #     )


class KLDivergenceLoss(nn.Module):
    """Wrapper class for KL Divergence loss. Only to be used for VAE models."""

    def __init__(self, model, loss_fn: str = "l1"):
        super(KLDivergenceLoss, self).__init__()
        self.model = model
        if loss_fn == "l1":
            self.construction_loss = l1_loss
        else:
            self.construction_loss = l2_loss

    def forward(self) -> None:
        """Construction loss + KL loss."""
        if hasattr(self.model, "get_kl_div"):
            kl_term = self.model.get_kl_div()
        else:
            kl_term = 0
        construction_loss = self.construction_loss(
            self.model.estimate, self.model.target
        )
        self.model.batch_loss = torch.add(construction_loss, kl_term)


class L1Loss(nn.Module):
    """Wrapper class for l1 loss."""

    def __init__(self, model):
        super(L1Loss, self).__init__()
        self.model = model

    def forward(self) -> None:
        self.model.batch_loss = l1_loss(self.model.estimate, self.model.target)


class L2Loss(nn.Module):
    """Wrapper class for l2 loss."""

    def __init__(self, model):
        super(L2Loss, self).__init__()
        self.model = model

    def forward(self) -> None:
        self.model.batch_loss = l2_loss(self.model.estimate, self.model.target)
        # gain_penalty = torch.linalg.norm(self.model.target) / torch.linalg.norm(self.model.estimate)
        # self.model.batch_loss = self.model.batch_loss + gain_penalty


# class SISDRLoss(nn.Module):
#     """Wrapper class for si-sdr loss."""
#
#     def __init__(self, model):
#         super(SISDRLoss, self).__init__()
#         self.model = model
#
#     def forward(self) -> None:
#         loss = scale_invariant_sdr_loss(self.model.estimate, self.model.target)
#         self.model.batch_loss = torch.mean(loss)
