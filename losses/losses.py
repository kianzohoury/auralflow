import torch
import torch.nn as nn
from torch.nn.functional import l1_loss


def kl_div_loss(mu: torch.FloatTensor, sigma: torch.FloatTensor) -> float:
    return torch.mean(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1) * 0.5


def vae_loss(estimate, target, mu, sigma):
    return l1_loss(estimate, target) + kl_div_loss(mu, sigma)