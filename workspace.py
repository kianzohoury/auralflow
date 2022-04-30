import torch

from config.build import build_layers
import models.modules
from models.modules import AutoEncoder2d, VAE2d
from pprint import pprint

if __name__ == '__main__':
    # d = build_layers({})
    autoencoder = AutoEncoder2d(
        num_targets=3,
        num_bins=512,
        num_samples=173,
        num_channels=1,
        max_depth=6,
        hidden_size=16,
        kernel_size=(3, 5, 3, 5),
        same_padding=True,
        block_size=3,
        downsampler='conv',
        upsampler='transpose',
        batch_norm=True,
        activation='leaky_relu',
        use_skip=True,
        dropout_p=0.5,
        leak_constant=0.2
    )
    vae = VAE2d(
        num_targets=3,
        latent_size=1024,
        num_bins=512,
        num_samples=173,
        num_channels=1,
        max_depth=6,
        hidden_size=16,
        kernel_size=(3, 5, 3, 5),
        same_padding=True,
        block_size=3,
        downsampler='conv',
        upsampler='transpose',
        batch_norm=True,
        activation='leaky_relu',
        use_skip=True,
        dropout_p=0.5,
        leak_constant=0.2
    )
    # autoencoder(torch.rand((1, 1, 512, 173)))
    vae(torch.rand((1, 1, 512, 173)))
