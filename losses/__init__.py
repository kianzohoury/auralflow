import torch
import torch.nn as nn
from torch.nn.functional import l1_loss


def kl_div_loss(mu: torch.FloatTensor, sigma: torch.FloatTensor) -> torch.Tensor:
    return torch.mean(torch.sum(mu**2 + sigma**2 - torch.log(sigma**2) - 1, dim=2) * 0.5)


def vae_loss(estimate, target, mu, sigma):
    return l1_loss(estimate, target) + kl_div_loss(mu, sigma)


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.kl = nn.KLDivLoss()

    def forward(self, net_est, target, latent_est, latent_target):
        print(net_est.shape, target.shape, latent_est.shape, latent_target.shape)
        return self.l1(net_est, target.squeeze(1)) + self.kl(latent_est, latent_target)
