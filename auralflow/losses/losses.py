# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import asteroid.metrics
import torch
import torch.nn as nn


from auralflow.models import SpectrogramMaskModel
from auralflow.transforms import trim_audio
from torch import FloatTensor, Tensor
from torch.nn import functional
from torchaudio.transforms import Resample
from typing import Mapping


def component_loss(
    mask: FloatTensor,
    target: FloatTensor,
    residual: FloatTensor,
    alpha: float = 0.2,
    beta: float = 0.8,
) -> Tensor:
    r"""Weighted loss that combines different loss components.

    Uses two or three loss components depending on the arguments for ``alpha``
    and ``beta``. If ``beta=0``, it weighs the quality of target source
    separation against the amount of residual noise attenuation. If ``beta>0``,
    the third component balances the quality of the attenuated residual noise
    against the other two terms. The 2-component loss is defined as:

    .. math::
        L_{2c}(X; Y_{k}; \theta; \alpha) = \frac{1-\alpha}{n} ||Y_{f, k}
            - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}

    and the 3-component loss is defined as:

    .. math::
        L_{3c}(X; Y_{k}; \theta; \alpha; \beta) = \frac{1-\alpha -\beta}{n}
            ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}
            + \frac{\beta}{n}|| \hat{R_f} - \hat{R}||_2^2

    where:
        - :math:`Y_{f, k} := M_{\theta}\odot |Y_{k}|`, filtered target
        - :math:`R_{f} := M_{ \theta }\odot (|X| - |Y_{k}|)`, filtered residual
        - :math:`\hat{R_{f}} := \frac{R_{f}}{||R_{f}||_2}`, filtered unit-residual
        - :math:`\hat{R} := \frac{R}{||R||_2}`, unit residual

    Args:
        mask (FloatTensor): Estimated soft-mask (output of the network).
        target (FloatTensor): Ground-truth target source.
        residual (FloatTensor): The residual or background noise, defined as
            the difference between the mixture and target sources.
        alpha (float): Value for the two-term component loss.
        beta (float): Value for the three-term component loss.

    Returns:
        Tensor: Mean component loss.

    Examples:
        >>> # get mask, target and residual spectrogram data
        >>> mask = torch.rand((16, 512, 173, 1)).float()
        >>> target = torch.rand((16, 512, 173, 1)).float()
        >>> residual = torch.rand((16, 512, 173, 1)).float()
        >>>
        >>> # calculate the weighted loss
        >>> loss = component_loss(mask=mask, target=target, residual=residual)
        >>> type(loss)
        <class 'torch.Tensor'>
        >>>
        >>> # get its scalar value
        >>> loss_val = loss.item()
        >>> type(loss_val)
        <class 'float'>
    """

    filtered_target = mask * target
    filtered_residual = mask * residual

    # Separation quality term.
    total_separation_loss = torch.sum((filtered_target - target) ** 2)

    # Noise attenuation term.
    total_noise_loss = torch.sum(filtered_residual ** 2)

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


def kl_div_loss(mu: FloatTensor, sigma: FloatTensor) -> Tensor:
    """Computes the KL Divergence loss term using its closed form expression.

    Only for ``SpectrogramNetVAE`` models. The KL loss term is defined
    as :math:`D_KL(P||Q)`, where :math:`P` is the modeled distribution,
    and :math:`Q` is a standard normal distribution :math:`N(0, 1)`.
    In closed form, the loss can be expressed as:

    .. math::
        D_{KL}(P||Q) = \\frac{1}{2} \\sum_{i=1}^{n}(\\mu^2 + \\sigma^2 -
            \\ln(\\sigma^2) - 1)

    where:
        - :math:`\\mu`: mean of the modeled distribution :math:`P`

        - :math:`\\sigma`: standard deviation of modeled distribution :math:`P`

        - :math:`n`: number of tensor elements

    Args:
        mu (FloatTensor): Mean of the modeled distribution.
        sigma (FloatTensor): Standard deviation of the modeled distribution.

    Returns:
        Tensor: Mean KL loss term.

    Examples:
        >>> # get mean and std data
        >>> mu = torch.zeros((16, 256, 256)).float()
        >>> sigma = torch.ones((16, 256, 256)).float()
        >>>
        >>> # get estimate and target spectrogram data
        >>> estimate_spec = torch.rand((16, 512, 173, 1))
        >>> target_spec = torch.rand((16, 512, 173, 1))
        >>>
        >>> # calculate kl div loss
        >>> kl_term = kl_div_loss(mu, sigma)
        >>>
        >>> # calculate reconstruction loss
        >>> l1_term = torch.nn.functional.l1_loss(estimate_spec, target_spec)
        >>>
        >>> # combine loss terms
        >>> loss = kl_term + l1_term
        >>> type(loss)
        <class 'torch.Tensor'>
        >>>
        >>> # get scalar value
        >>> loss_val = loss.item()
        >>> type(loss_val)
        <class 'float'>
    """
    return 0.5 * torch.mean(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1)


def si_sdr_loss(estimate: FloatTensor, target: FloatTensor) -> Tensor:
    """Scale-invariant (SI-SDR) signal to distortion loss.

    Typically used as an evaluation metric, but can be optimized for directly.
    For a single audio track, the loss is defined as:

    .. math::
        L_{\\text{SI-SDR}}(\\hat y, y) = -10\\log_{10} \\frac{||\\frac{proj_{y}
            \\hat y}{||y||_2^2}||_2^2}{||\\frac{proj_{y} \\hat y}{||y||_2^2}
            - \\hat y||_2^2}

    where:
        - :math:`y`: true target signal
        - :math:`\\hat y`: estimated target source signal

    Args:
        estimate (FloatTensor): Target source estimate in the audio domain.
        target (FloatTensor): Ground-truth target source in the audio domain.

    Returns:
        Tensor: Mean SI-SDR loss.

    Examples:
        >>> # get estimate and target audio data
        >>> estimate = torch.rand((16, 1, 88200))
        >>> target = torch.rand((16, 1, 88200))
        >>>
        >>> # calculate SI-SDR loss
        >>> loss = si_sdr_loss(estimate, target)
        >>> type(loss)
        <class 'torch.Tensor'>
        >>>
        >>> # get its scalar value
        >>> loss_val = loss.item()
        >>> type(loss_val)
        <class 'float'>
    """
    # Optimal scaling factor alpha.
    top_term = torch.sum(estimate * target, dim=-1, keepdim=True)
    bottom_term = torch.sum(target ** 2, dim=-1, keepdim=True)
    alpha = top_term / bottom_term

    # Target signal error term.
    error_target = alpha * target
    # Residual/noise signal error term.
    error_residual = estimate - error_target

    # Numerator term.
    signal_term = torch.sum(error_target ** 2, dim=(1, 2))
    distortion_term = torch.sum(error_residual ** 2, dim=(1, 2))

    # Loss.
    loss = -torch.mean(10 * torch.log10(signal_term / distortion_term), dim=0)
    return loss


def rmse_loss(
        estimate: FloatTensor, target: Tensor, eps: float = 1e-6
) -> Tensor:
    """RMSE loss."""
    return torch.sqrt(functional.mse_loss(estimate, target) + eps)


class ComponentLoss(nn.Module):
    """Wrapper class for ``component_loss``.

    Uses two or three loss components depending on the arguments for ``alpha``
    and ``beta``. If ``beta=0``, it weighs the quality of target source
    separation against the amount of residual noise attenuation. If ``beta>0``,
    the third component balances the quality of the attenuated residual noise
    against the other two terms. The 2-component loss is defined as:

    Note:
        Sets ``batch_loss`` attribute of the model instead of returning the
        loss directly.

    .. math::
        L_{2c}(X; Y_{k}; \theta; \alpha) = \frac{1-\alpha}{n} ||Y_{f, k}
            - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}

    and the 3-component loss is defined as:

    .. math::
        L_{3c}(X; Y_{k}; \theta; \alpha; \beta) = \frac{1-\alpha -\beta}{n}
            ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}
            + \frac{\beta}{n}|| \\hat{R_f} - \\hat{R}||_2^2

    where:
        - :math:`Y_{f, k} := M_{\theta}\\odot |Y_{k}|`, filtered target
        - :math:`R_{f} := M_{ \theta }\\odot (|X| - |Y_{k}|)`, filtered residual
        - :math:`\\hat{R_{f}} := \\frac{R_{f}}{||R_{f}||_2}`, filtered unit-residual
        - :math:`\\hat{R} := \\frac{R}{||R||_2}`, unit residual

    Args:
        model (SpectrogramMaskModel): Spectrogram mask model.
        alpha (float): Value for the two-term component loss.
        beta (float): Value for the three-term component loss.
    """

    def __init__(
        self, model: SpectrogramMaskModel, alpha: float, beta: float
    ) -> None:
        super(ComponentLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta

    def forward(self) -> None:
        """Calculates weighted component loss and stores it."""
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


class KLDivergenceLoss(nn.Module):
    """Wrapper class for ``kl_div_loss``.

    The KL loss term is defined as :math:`D_KL(P||Q)`, where :math:`P` is
    the modeled distribution, and :math:`Q` is a standard normal distribution
    :math:`N(0, 1)`. In closed form, the loss can be expressed as:

    .. math::
        D_{KL}(P||Q) = \\frac{1}{2} \\sum_{i=1}^{n}(\\mu^2 + \\sigma^2 -
            \\ln(\\sigma^2) - 1)

    where:
        - :math:`\\mu`: mean of the modeled distribution :math:`P`

        - :math:`\\sigma`: standard deviation of modeled distribution :math:`P`

        - :math:`n`: number of tensor elements

    Note:
        Sets ``batch_loss`` attribute of the model instead of returning the
        loss directly. Can also only be used with ``SpectrogramNetVAE``.

    Args:
        model (SpectrogramMaskModel): Spectrogram mask model.
        loss_fn (str): Construction loss criterion. Default: 'l1'.
    """

    def __init__(
        self, model: SpectrogramMaskModel, loss_fn: str = "l1"
    ) -> None:
        super(KLDivergenceLoss, self).__init__()
        self.model = model
        if loss_fn == "l1":
            self.construction_loss = nn.functional.l1_loss
        else:
            self.construction_loss = nn.functional.mse_loss

    def forward(self) -> None:
        """Calculates construction loss + KL loss and stores it."""
        if hasattr(self.model, "get_kl_div"):
            kl_term = kl_div_loss(
                mu=self.model.model.mu_data, sigma=self.model.model.sigma_data
            )
        else:
            kl_term = 0
        construction_loss = self.construction_loss(
            self.model.estimate, self.model.target
        )
        self.model.batch_loss = construction_loss + kl_term


class SISDRLoss(nn.Module):
    """Wrapper class for ``si_sdr_loss``.

    Typically used as an evaluation metric, but can be optimized for directly.
    For a single audio track, the loss is defined as:

    .. math::
        L_{\\text{SI-SDR}}(\\hat y, y) = -10\\log_{10} \\frac{||\\frac{proj_{y}
            \\hat y}{||y||_2^2}||_2^2}{||\\frac{proj_{y} \\hat y}{||y||_2^2}
            - \\hat y||_2^2}

    where:
        - :math:`y`: true target signal
        - :math:`\\hat y`: estimated target source signal

    Note:
        Sets ``batch_loss`` attribute of the model instead of returning the
        loss directly.

    Args:
        model (SpectrogramMaskModel): Spectrogram mask model.
    """

    def __init__(self, model) -> None:
        super(SISDRLoss, self).__init__()
        self.model = model

    def forward(self) -> None:
        """Calculates si_sdr_loss and stores it."""
        estimate_audio = self.model.estimate_audio[..., :self.model.target_audio.shape[-1]]
        target_audio = self.model.target_audio.squeeze(-1).float()
        self.model.batch_loss = si_sdr_loss(
            estimate_audio, target_audio
        )


class L1Loss(nn.Module):
    """Wrapper class for l1 loss.

    Args:
        model (SpectrogramMaskModel): Spectrogram mask model.

    Note:
        Sets ``batch_loss`` attribute of the model instead of returning the
        loss directly.
    """

    def __init__(self, model: SpectrogramMaskModel) -> None:
        super(L1Loss, self).__init__()
        self.model = model

    def forward(self) -> None:
        """Calculates l1 loss and stores it."""
        self.model.batch_loss = nn.functional.l1_loss(
            self.model.estimate, self.model.target
        )


class L2Loss(nn.Module):
    """Wrapper class for l2 loss.

    Args:
        model (SpectrogramMaskModel): Spectrogram mask model.

    Note:
        Sets ``batch_loss`` attribute of the model instead of returning the
        loss directly.
    """

    def __init__(self, model: SpectrogramMaskModel) -> None:
        super(L2Loss, self).__init__()
        self.model = model

    def forward(self) -> None:
        """Calculates l2 loss and stores it."""
        self.model.batch_loss = nn.functional.mse_loss(
            self.model.estimate, self.model.target
        )


class RMSELoss(nn.Module):
    """Wrapper class for rmse loss."""

    def __init__(self, model):
        super(RMSELoss, self).__init__()
        self.model = model

    def forward(self) -> None:
        sep_loss = rmse_loss(self.model.estimate, self.model.target)
        mask_loss = rmse_loss(self.model.mix_phase, self.model.target_phase)
        self.model.batch_loss = 0.5 * sep_loss + 0.5 * mask_loss


class L2MaskLoss(nn.Module):
    """Wrapper class for l2 loss directly on masks."""

    def __init__(self, model):
        super(L2MaskLoss, self).__init__()
        self.model = model

    def forward(self) -> None:
        ideal_mask = self.model.target / torch.max(
            self.model.mixture, torch.ones_like(self.model.mixture)
        )
        self.model.batch_loss = nn.functional.mse_loss(
            self.model.mask, ideal_mask
        )


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

