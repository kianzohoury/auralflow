from typing import Mapping, OrderedDict

import torch
import torch.nn as nn
import numpy as np


from torch import FloatTensor, Tensor
from torch.nn import functional
from nussl.evaluation import scale_bss_eval
from utils.data_utils import trim_audio


eval_metrics_labels = [
    "si-sdr",
    "si-sir",
    "si-sar",
    "sd-sdr",
    "snr",
    "srr",
    "si-sdri",
    "sd-sdri",
    "snri"
]


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


def get_evaluation_metrics(
    mix: Tensor, estimate: FloatTensor, target: Tensor, full: bool = True
) -> OrderedDict[str, float]:
    """Returns the batch-wise mean standardized source separation scores."""
    # Collapse channels dimension to mono.
    mix = torch.mean(mix, dim=1).unsqueeze(-1).cpu().numpy()
    estimate = torch.mean(estimate, dim=1).unsqueeze(-1).cpu().numpy()
    target = torch.mean(target, dim=1).unsqueeze(-1).cpu().numpy()
    print(mix.shape, estimate.shape, target.shape)
    scores = []

    # Compute scores for each sample.
    for i in range(mix.shape[0]):
        scores.append(
            scale_bss_eval(
                references=target[0],
                estimate=estimate[0],
                mixture=mix[0],
                idx=0,
                compute_sir_sar=full
            )
        )

    # Average scores.
    avg_scores = np.mean(scores, axis=0, keepdims=False)
    if not full:
        labels = eval_metrics_labels[:1] + eval_metrics_labels[3:]
    else:
        labels = eval_metrics_labels
    named_metrics = {
        labels[i]: avg_scores[i] for i in range(len(avg_scores))
    }
    return named_metrics


class WeightedComponentLoss(nn.Module):
    """Wrapper class for calling weighted component loss."""

    def __init__(self, model, alpha: float, beta: float):
        super(WeightedComponentLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta

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


class SeparationEvaluator(object):
    """Wrapper class for computing evaluation metrics."""
    def __init__(self, model, full_metrics: bool = True):
        super(SeparationEvaluator, self).__init__()
        self.model = model
        self.full_metrics = full_metrics

    def get_metrics(
        self, mix: Tensor, target: Tensor
    ) -> OrderedDict[str, float]:
        """Returns evaluation metrics."""
        estimate = self.model.separate(mix.squeeze(-1).to(self.model.device))
        mix, estimate, target = trim_audio(
            [mix.squeeze(-1), estimate, target.squeeze(-1)]
        )
        eval_metrics = get_evaluation_metrics(
            mix=mix, estimate=estimate, target=target, full=self.full_metrics
        )
        return eval_metrics

    @staticmethod
    def print_metrics(metrics: OrderedDict[str, float]) -> None:
        for label, value in metrics.items():
            print(f"{label}: {value}")
