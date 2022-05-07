import torch
import torch.nn as nn


def kl_div_loss(
    mu: torch.FloatTensor, sigma: torch.FloatTensor
) -> torch.Tensor:
    """Computes KL term using the closed form expression."""
    return 0.5 * torch.mean(mu**2 + sigma**2 - torch.log(sigma**2) - 1)


def vae_loss(const_criterion, estimate, target, kl_term):
    """Computes VAE loss := construction_loss(x, x') + kl_loss(P, Q)"""
    return const_criterion(estimate, target) + kl_term


class KLDivergenceLoss(nn.Module):
    """Implements KL Divergence loss with native PyTorch implementation."""

    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.kl = nn.KLDivLoss()

    def forward(self, net_est, target, latent_est, latent_target):
        print(
            net_est.shape, target.shape, latent_est.shape, latent_target.shape
        )
        return self.l1(net_est, target.squeeze(1)) + self.kl(
            latent_est, latent_target
        )
