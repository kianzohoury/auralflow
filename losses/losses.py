import torch
import torch.nn as nn


def kl_div_loss(mu: torch.FloatTensor, sigma: torch.FloatTensor) -> float:
    return torch.sum(mu**2 + sigma**-1 - torch.log(sigma**2)) * 0.5
