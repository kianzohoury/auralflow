# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch
import torch.nn as nn


from fast_bss_eval import si_sdr_loss as si_sdr_loss_
from torch import FloatTensor, Tensor
from torch.nn import functional
from typing import Mapping
from auralflow.utils.data_utils import trim_audio


# eval_metrics_labels = [
#     "si-sdr",
#     "si-sir",
#     "si-sar",
#     "sd-sdr",
#     "snr",
#     "srr",
#     "si-sdri",
#     "sd-sdri",
#     "snri",
# ]


def component_loss(
    filtered_src: FloatTensor,
    target_src: Tensor,
    filtered_res: FloatTensor,
    target_res: Tensor,
    alpha: float = 0.2,
    beta: float = 0.8,
    n_components: int = 2,
) -> Tensor:
    """Weighted L2 loss using 2 or 3 components depending on arguments.

    Balances the target source separation quality versus the amount of
    residual noise attenuation. Optional third component balances the
    quality of the residual noise against the other two terms.
    """
    # Separation quality term.
    total_separation_loss = torch.sum((filtered_src - target_src) ** 2)

    # Noise attenuation term.
    total_noise_loss = torch.sum(filtered_res**2)

    summed_dims = list(range(1, filtered_res.dim()))
    filtered_res_unit = filtered_res / torch.linalg.norm(
        target_res, dim=summed_dims, keepdim=True
    )
    target_res_unit = target_res / torch.linalg.norm(
        target_res, dim=summed_dims, keepdim=True
    )

    # Noise quality term.
    total_noise_quality_loss = torch.sum(
        (filtered_res_unit - target_res_unit) ** 2
    )

    # Discards last term if specified.
    beta = 0 if n_components == 2 else beta

    # Constrain alpha + beta <= 1.
    if alpha + beta > 1:
        total = alpha + beta + 1e-8
        alpha = alpha / total
        beta = beta / total

    # Combine loss components.
    loss = (1 - alpha - beta) * total_separation_loss
    loss = loss + alpha * total_noise_loss
    loss = loss + beta * total_noise_quality_loss
    mean_loss = loss / filtered_src.numel()
    return mean_loss


def l1_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """L1 loss."""
    return functional.l1_loss(estimate, target)


def l2_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """L2 loss."""
    return functional.mse_loss(estimate, target)


def kl_div_loss(mu: FloatTensor, sigma: FloatTensor) -> Tensor:
    """Computes KL term using the closed form expression."""
    return 0.5 * torch.mean(mu**2 + sigma**2 - torch.log(sigma**2) - 1)


def scale_invariant_sdr_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """Computes scale-invariant signal-distortion ratio loss."""
    return si_sdr_loss_(est=estimate, ref=target, zero_mean=True, clamp_db=1)


def get_evaluation_metrics(
    estimate: FloatTensor, target: Tensor
) -> Mapping[str, float]:
    """Returns batch-wise means of standard source separation eval scores."""
    estimate = torch.mean(estimate, dim=1, keepdim=True)
    target = torch.mean(target, dim=1, keepdim=True)

    # Scale-invariant signal-distortion ratio.
    si_sdr_ = si_sdr_loss_(ref=target, est=estimate, zero_mean=True)
    named_metrics = {"si_sdr": torch.mean(si_sdr_).item()}

    return named_metrics


class WeightedComponentLoss(nn.Module):
    """Wrapper class for calling weighted component loss."""

    def __init__(
        self, model, alpha: float, beta: float, regularizer: bool = True
    ):
        super(WeightedComponentLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer
        # if regularizer:
        #     self.reg_loss = l2_loss

    def forward(self):
        """Calculates a weighted component loss."""
        # Apply mask to true target source.
        filtered_src = self.model.mask * self.model.target

        # Apply mask to true residual.
        true_residual = self.model.mixture - self.model.target
        filtered_res = self.model.mask * true_residual

        # Compute weighted loss.
        self.model.batch_loss = component_loss(
            filtered_src=filtered_src,
            target_src=self.model.target,
            filtered_res=filtered_res,
            target_res=true_residual,
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
    """Wrapper class for KL Divergence loss. Only to be used for VAE models.

    KL term is defined as := D_KL(P||Q), where P is the modeled distribution,
    and Q is a standard normal N(0, 1). The term is combined with the
    reconstruction loss.
    """

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


class SISDRLoss(nn.Module):
    """Wrapper class for si-sdr loss."""

    def __init__(self, model):
        super(SISDRLoss, self).__init__()
        self.model = model

    def forward(self) -> None:
        loss = scale_invariant_sdr_loss(self.model.estimate, self.model.target)
        self.model.batch_loss = torch.mean(loss)


class SeparationEvaluator(object):
    """Wrapper class for computing evaluation metrics."""

    def __init__(self, model, full_metrics: bool = True):
        super(SeparationEvaluator, self).__init__()
        self.model = model
        self.full_metrics = full_metrics

    def get_metrics(self, mix: Tensor, target: Tensor) -> Mapping[str, float]:
        """Returns evaluation metrics."""
        estimate = self.model.separate(mix.squeeze(-1))
        estimate, target = trim_audio([estimate, target.squeeze(-1)])
        eval_metrics = get_evaluation_metrics(
            estimate=estimate, target=target.to(estimate.device)
        )
        return eval_metrics

    @staticmethod
    def print_metrics(metrics: Mapping[str, float]) -> None:
        for label, value in metrics.items():
            print(f"{label}: {value}")
