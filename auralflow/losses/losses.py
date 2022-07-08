# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import asteroid.metrics
import torch
import torch.nn as nn


from auralflow.models import SpectrogramMaskModel, SpectrogramNetVAE, SeparationModel
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
        - :math:`n`: number of tensor elements
        - :math:`\alpha`: 2-term coefficient value
        - :math:`\alpha`: 3-term coefficient value

    Args:
        mask (FloatTensor): Estimated soft-mask (output of the network).
        target (FloatTensor): Ground-truth target source.
        residual (FloatTensor): The residual or background noise, defined as
            the difference between the mixture and target sources.
        alpha (float): Coefficient value for the two-term component loss.
            Default: ``0.2``.
        beta (float): Coefficient value for the three-term component loss.
            Default: ``0.8``.

    Returns:
        Tensor: Mean component loss.

    Examples:

        Get mask, target and resiual spectrograms:

        >>> import torch
        >>> mask = torch.rand((16, 512, 173, 1)).float()
        >>> target = torch.rand((16, 512, 173, 1)).float()
        >>> residual = torch.rand((16, 512, 173, 1)).float()

        Calculate the weighted loss:

        >>> loss = component_loss(mask=mask, target=target, residual=residual)
        >>> type(loss)
        <class 'torch.Tensor'>

        Get its scalar value:

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

        Get mean and standard deviation of the latent distribution:

        >>> import torch
        >>> mu = torch.zeros((16, 256, 256)).float()
        >>> sigma = torch.ones((16, 256, 256)).float()

        Calculate KL loss:

        >>> loss = kl_div_loss(mu, sigma)
        >>> type(loss)
        <class 'torch.Tensor'>

        Get its scalar value:

        >>> loss_val = loss.item()
        >>> type(loss_val)
        <class 'float'>
    """
    return 0.5 * torch.mean(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1)


def si_sdr_loss(estimate: FloatTensor, target: FloatTensor) -> Tensor:
    """Scale-invariant (SI-SDR) signal-to-distortion loss.

    Typically used as an evaluation metric, but can be optimized for directly.
    For a single audio track, the loss is defined as:

    .. math::
        L_{\\text{SI-SDR}}(\\hat y, y) = -10\\log_{10} \\frac{||\\frac{proj_{y}
            \\hat y}{||y||_2^2}||_2^2}{||\\frac{proj_{y} \\hat y}{||y||_2^2}
            - \\hat y||_2^2}

    where:
        - :math:`y`: ground-truth target source signal
        - :math:`\\hat y`: estimated target source signal

    Args:
        estimate (FloatTensor): Source estimate signal data.
        target (FloatTensor): Ground-truth target source signal data.

    Returns:
        Tensor: Mean SI-SDR loss.

    Examples:

        Get estimate and target audio signals:

        >>> import torch
        >>> estimate = torch.rand((16, 1, 88200))
        >>> target = torch.rand((16, 1, 88200))

        Calculate SI-SDR loss:

        >>> loss = si_sdr_loss(estimate, target)
        >>> type(loss)
        <class 'torch.Tensor'>

        Get its scalar value:

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


# def rmse_loss(
#     estimate: FloatTensor, target: Tensor, eps: float = 1e-6
# ) -> Tensor:
#     """RMSE loss."""
#     return torch.sqrt(functional.mse_loss(estimate, target) + eps)


class ComponentLoss(nn.Module):
    r"""Wrapper class for ``component_loss``.

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
        - :math:`n`: number of tensor elements
        - :math:`\alpha`: 2-term coefficient value
        - :math:`\alpha`: 3-term coefficient value

    Args:
        alpha (float): Value for the two-term component loss. Default: ``0.2``.
        beta (float): Value for the three-term component loss.
            Default: ``0.8``.

    Note:
        If ``model.model`` is of type ``SpectrogramNetVAE``, the result of
        ``kl_div_loss`` will be added to the loss.
    """

    def __init__(self, alpha: float = 0.2, beta: float = 0.8) -> None:
        super(ComponentLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        mask: FloatTensor,
        mix_spec: FloatTensor,
        target_spec: FloatTensor,
    ) -> Tensor:
        """Calculates the weighted component loss.

        Returns:
            Tensor: Weighted component loss.
        """
        # Compute weighted loss.
        loss = component_loss(
            mask=mask,
            target=target_spec,
            residual=mix_spec - target_spec,
            alpha=self.alpha,
            beta=self.beta,
        )
        return loss

    def _forward_wrapper(
        self,
        model: SpectrogramMaskModel,
        mix_audio: Tensor,
        target_audio: Tensor
    ) -> Tensor:
        """Wraps a model's full forward pass.

        Calls ``self.forward(...)`` after ``model.forward(...)``.
        """
        # Preprocess data.
        mix_spec, mix_phase = model.to_spectrogram(audio=mix_audio)
        target_spec, _ = model.to_spectrogram(audio=target_audio)
        # Run model's forward pass.
        estimate_spec, data = model.forward(mixture=mix_spec)
        # Get loss.
        loss = self.forward(
            mask=data["mask"],
            mix_spec=mix_spec,
            target_spec=target_spec
        )
        # Add KL loss term if applicable.
        if isinstance(model.model, SpectrogramNetVAE):
            mu, sigma = data["mu"], data["sigma"]
            kl_loss = kl_div_loss(mu=mu, sigma=sigma)
            loss = loss + kl_loss
        return loss


class KLDivergenceLoss(nn.Module):
    r"""Wrapper class for ``kl_div_loss``.

    The KL loss term is defined as
    :math:`D_KL(P||Q)`, where :math:`P` is the modeled distribution, and
    :math:`Q` is a standard normal distribution :math:`N(0, 1)`. In closed
    form, the loss can be expressed as:

    .. math::

        D_{KL}(P||Q) = \frac{1}{2} \sum_{i=1}^{n}(\mu^2 + \sigma^2 -
            \ln(\sigma^2) - 1)

    where:
        - :math:`\mu`: mean of the modeled distribution :math:`P`

        - :math:`\sigma`: standard deviation of modeled distribution :math:`P`

        - :math:`n`: number of tensor elements

    Args:
        loss_fn (str): Construction loss criterion. Default: ``'l1'``.

    Note:
        Can only be used if ``model.model`` is an instance of
        ``SpectrogramNetVAE``.
    """

    def __init__(self, loss_fn: str = "l1") -> None:
        super(KLDivergenceLoss, self).__init__()
        if loss_fn == "l1":
            self.construction_loss = nn.functional.l1_loss
        else:
            self.construction_loss = nn.functional.mse_loss

    def forward(
        self,
        estimate_spec: FloatTensor,
        target_spec: FloatTensor,
        mu: FloatTensor,
        sigma: FloatTensor
    ) -> Tensor:
        """Adds together the reconstruction loss and KL loss.

        Returns:
            Tensor: Combined loss.
        """
        kl_term = kl_div_loss(mu=mu, sigma=sigma)
        construction_loss = self.construction_loss(estimate_spec, target_spec)
        loss = construction_loss + kl_term
        return loss

    def _forward_wrapper(
        self,
        model: SpectrogramMaskModel,
        mix_audio: Tensor,
        target_audio: Tensor
    ) -> Tensor:
        """Wraps a model's full forward pass.

        Calls self.forward(...) after model.forward(...). Used in
        ``trainer.run_training(...)`` to fully encapsulate the forward phase.
        """
        if not isinstance(model.model, SpectrogramNetVAE):
            raise ValueError(
                f"Expected model of type {type(SpectrogramNetVAE)}, but "
                f"received {type(model.model)}."
            )
        # Preprocess data.
        mix_spec, mix_phase = model.to_spectrogram(audio=mix_audio)
        target_spec, _ = model.to_spectrogram(audio=target_audio)
        # Run model's forward pass.
        estimate_spec, data = model.forward(mixture=mix_spec)
        # Get loss.
        mu, sigma = data["mu"], data["sigma"]
        loss = self.forward(
            estimate_spec=estimate_spec,
            target_spec=target_spec,
            mu=mu,
            sigma=sigma
        )
        return loss


class SISDRLoss(nn.Module):
    r"""Wrapper class for ``si_sdr_loss``.

    Typically used as an evaluation metric, but can be optimized for directly.
    For a single audio track, the loss is defined as:

    .. math::

        L_{\text{SI-SDR}}(\hat y, y) = -10\cdot\log_{10}\frac{||\frac{proj_{y}
            \hat y}{||y||_2^2}||_2^2}{||\frac{proj_{y} \hat y}{||y||_2^2}
            - \hat y||_2^2}

    where:
        - :math:`y`: ground-truth target source signal
        - :math:`\hat y`: estimated target source signal
    """

    def __init__(self, best_perm: bool = True) -> None:
        super(SISDRLoss, self).__init__()
        self.best_perm = best_perm

    def forward(self, estimate_audio: Tensor, target_audio: Tensor) -> Tensor:
        """Calculates signal-to-distortion ratio loss.

        Returns:
            Tensor: SI-SDR loss.
        """
        estimate_audio = estimate_audio[..., :target_audio.shape[-1]]
        target_audio = target_audio.squeeze(-1)
        loss = si_sdr_loss(
            estimate=estimate_audio, target=target_audio
        )
        if self.best_perm:
            loss_perm = si_sdr_loss(
                estimate=target_audio, target=estimate_audio
            )
            loss = min(loss, loss_perm, key=lambda x: x.item())
        return loss

    def _forward_wrapper(
        self,
        model: SpectrogramMaskModel,
        mix_audio: Tensor,
        target_audio: Tensor
    ) -> Tensor:
        """Wraps a model's full forward pass.

        Calls self.forward(...) after model.forward(...). Used in
        ``trainer.run_training(...)`` to fully encapsulate the forward phase.
        """
        # Preprocess data.
        mix_spec, mix_phase = model.to_spectrogram(audio=mix_audio)
        # Run model's forward pass.
        estimate_spec, data = model.forward(mixture=mix_spec)
        # Separate.
        estimate_audio = model.separate(mixture=estimate_spec)
        target_audio = target_audio.to(model.device)
        # Get loss.
        loss = self.forward(
            estimate_audio=estimate_audio, target_audio=target_audio
        )
        # Add KL loss term if applicable.
        if isinstance(model.model, SpectrogramNetVAE):
            mu, sigma = data["mu"], data["sigma"]
            kl_loss = kl_div_loss(mu=mu, sigma=sigma)
            loss = loss + kl_loss
        return loss


class L1Loss(nn.Module):
    """Wrapper class for l1 loss.

    Args:
        reduce_mean (bool): If ``True``, returns the batch-wise mean of the
            loss; otherwise, returns the batch-wise sum of the loss.
    """

    def __init__(self, reduce_mean: bool = True) -> None:
        super(L1Loss, self).__init__()
        self.reduction = "mean" if reduce_mean else "sum"

    def forward(
        self, estimate_spec: FloatTensor, target_spec: FloatTensor
    ) -> Tensor:
        """Calculates the l1 loss.

        Returns:
            Tensor: L1 loss.
        """
        loss = nn.functional.l1_loss(
            estimate_spec, target_spec, reduction=self.reduction)
        return loss

    def _forward_wrapper(
        self,
        model: SpectrogramMaskModel,
        mix_audio: Tensor,
        target_audio: Tensor
    ) -> Tensor:
        """Wraps a model's full forward pass.

        Calls self.forward(...) after model.forward(...). Used in
        ``trainer.run_training(...)`` to fully encapsulate the forward phase.
        """
        # Preprocess data.
        mix_spec, mix_phase = model.to_spectrogram(audio=mix_audio)
        target_spec, _ = model.to_spectrogram(audio=target_audio)
        # Run model's forward pass.
        estimate_spec, _ = model.forward(mixture=mix_spec)
        # Get loss.
        loss = self.forward(
            estimate_spec=estimate_spec, target_spec=target_spec
        )
        return loss


class L2Loss(nn.Module):
    """Wrapper class for l2 loss.

    Args:
        reduce_mean (bool): If ``True``, returns the batch-wise mean of the
            loss; otherwise, returns the batch-wise sum of the loss.
    """

    def __init__(self, reduce_mean: bool = True) -> None:
        super(L2Loss, self).__init__()
        self.reduction = "mean" if reduce_mean else "sum"

    def forward(
        self, estimate_spec: FloatTensor, target_spec: FloatTensor
    ) -> Tensor:
        """Calculates the l2 loss.

        Returns:
            Tensor: L2 loss.
        """
        loss = nn.functional.mse_loss(
            estimate_spec, target_spec, reduction=self.reduction)
        return loss

    def _forward_wrapper(
        self,
        model: SpectrogramMaskModel,
        mix_audio: Tensor,
        target_audio: Tensor
    ) -> Tensor:
        """Wraps a model's full forward pass.

        Calls self.forward(...) after model.forward(...). Used in
        ``trainer.run_training(...)`` to fully encapsulate the forward phase.
        """
        # Preprocess data.
        mix_spec, mix_phase = model.to_spectrogram(audio=mix_audio)
        target_spec, _ = model.to_spectrogram(audio=target_audio)
        # Run model's forward pass.
        estimate_spec, _ = model.forward(mixture=mix_spec)
        # Get loss.
        loss = self.forward(
            estimate_spec=estimate_spec, target_spec=target_spec
        )
        return loss

#
# class RMSELoss(nn.Module):
#     """Wrapper class for rmse loss."""
#
#     def __init__(self):
#         super(RMSELoss, self).__init__()
#         self.model = model
#
#     def forward(self) -> None:
#         sep_loss = rmse_loss(self.model.estimate_spec, self.model.target_spec)


class MaskLoss(nn.Module):
    r"""Wrapper class for calculating the loss on soft-masks directly.

    The loss is defined as:

    .. math::

        L_{mask}(X, Y, M_{\theta}) = l(\frac{|Y|}{max(|X|, \epsilon)}, M_{\theta})

    where :math:`l` is the loss function, :math:`|X|` and :math:`|Y|` are
    the magnitude spectrograms of the mixture and targets respectively,
    :math:`M_{\theta}` is the estimated soft-mask (output of the network) and
    :math:`\epsilon` is a small constant added for numerical stability.

    Args:
        loss_fn (str): Loss function. Default: ``'l1'``.
        reduce_mean (bool): If ``True``, returns the batch-wise mean of the
            loss; otherwise, returns the batch-wise sum of the loss.
    """

    def __init__(self, loss_fn="l1", reduce_mean: bool = True) -> None:
        super(MaskLoss, self).__init__()
        if loss_fn == "l1":
            self.criterion = nn.functional.l1_loss
        else:
            self.criterion = nn.functional.mse_loss
        self.reduction = "mean" if reduce_mean else "sum"

    def forward(
        self,
        mask: FloatTensor,
        mix_spec: FloatTensor,
        target_spec: FloatTensor,
        eps: float = 1e-6
    ) -> Tensor:
        """Calculates the mask loss.

        Returns:
            Tensor: Mask loss.
        """
        ideal_mask = target_spec / torch.max(
            mix_spec, torch.full_like(
                mix_spec, fill_value=eps, device=mix_spec.device
            )
        )
        loss = self.criterion(mask, ideal_mask, reduction=self.reduction)
        return loss

    def _forward_wrapper(
        self,
        model: SpectrogramMaskModel,
        mix_audio: Tensor,
        target_audio: Tensor
    ) -> Tensor:
        """Wraps a model's full forward pass.

        Calls self.forward(...) after model.forward(...). Used in
        ``trainer.run_training(...)`` to fully encapsulate the forward phase.
        """
        # Preprocess data.
        mix_spec, mix_phase = model.to_spectrogram(audio=mix_audio)
        target_spec, _ = model.to_spectrogram(audio=target_audio)
        # Run model's forward pass.
        estimate_spec, data = model.forward(mixture=mix_spec)
        # Get loss.
        loss = self.forward(
            mask=data["mask"], mix_spec=mix_spec, target_spec=target_spec
        )
        # Add KL loss term if applicable.
        if isinstance(model.model, SpectrogramNetVAE):
            mu, sigma = data["mu"], data["sigma"]
            kl_loss = kl_div_loss(mu=mu, sigma=sigma)
            loss = loss + kl_loss
        return loss



















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

