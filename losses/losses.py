import torch
import torch.nn as nn

from torch import FloatTensor, Tensor
from torch.nn import functional


def component_loss(
    filtered_src: FloatTensor,
    target_src: Tensor,
    filtered_res: FloatTensor,
    target_res: Tensor,
    alpha: float = 0.1,
    beta: float = 0.8,
    n_components: int = 3
) -> Tensor:
    """Weighted L2 loss using 2 or 3 components depending on arguments.

    Balances the target source separation quality versus the amount of
    residual noise attenuation. Optional third component balances the
    quality of the residual noise against the other two terms.
    """
    # Separation quality term. Measures the quality of the estimated target.
    sep_comp = ((filtered_src - target_src) ** 2).sum()

    # Noise attenuation term. Measures the total noise of the residual.
    noise_atten_comp = (filtered_res ** 2).sum()

    # Noise quality component.
    filtered_res_norm = filtered_res / torch.sqrt((filtered_res ** 2).sum())
    target_res_norm = target_res / torch.sqrt((target_res ** 2).sum())
    noise_quality_comp = ((filtered_res_norm - target_res_norm) ** 2).sum()

    # Discards last term if specified.
    beta = 0 if n_components == 2 else beta

    # Constrain alpha + beta <= 1.
    if alpha + beta > 1:
        alpha /= (alpha + beta + 1e-6)
        beta /= (alpha + beta + 1e-6)

    # Combine loss components.
    loss = (1 - alpha - beta) * sep_comp
    loss += alpha * noise_atten_comp
    loss += beta * noise_quality_comp
    return loss


def l1_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """L1 loss."""
    return functional.l1_loss(estimate, target)


def l2_loss(estimate: FloatTensor, target: Tensor) -> Tensor:
    """L2 loss."""
    return functional.mse_loss(estimate, target)


def kl_div_loss(mu: FloatTensor, sigma: FloatTensor) -> Tensor:
    """Computes KL term using the closed form expression."""
    return 0.5 * torch.mean(mu**2 + sigma**2 - torch.log(sigma**2) - 1)


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
        filtered_src = self.model.mask * self.model.target.squeeze(-1)

        # Apply mask to true residual.
        filtered_res = self.model.mask * (
            self.model.mixture - self.model.target.squeeze(-1)
        )

        # Compute weighted loss.
        self.model.batch_loss = component_loss(
            filtered_src=filtered_src,
            target_src=self.model.target.squeeze(-1),
            filtered_res=filtered_res,
            target_res=self.model.mixture - self.model.target.squeeze(-1),
            alpha=self.alpha,
            beta=self.beta
        )

        # Add kl term if using VAE.
        if hasattr(self.model, "get_kl_div"):
            torch.add(self.model.batch_loss, self.model.get_kl_div())


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
