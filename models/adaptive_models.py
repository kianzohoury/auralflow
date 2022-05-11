import inspect

import torch
import torch.nn as nn

from torch.nn import L1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from validate import cross_validate
from validate import cross_validate

from typing import Union, Tuple, Any, List, Callable
from .modules import AutoEncoder2d
from .layers import _get_activation
from .base import SeparationModel
from .architectures import (
    SpectrogramNetSimple,
    SpectrogramLSTM,
    SpectrogramLSTMVariational,
)
from visualizer import visualize_audio
from losses import vae_loss
from utils.data_utils import get_num_frames, get_stft, get_inverse_stft
from torch import Tensor, FloatTensor


class TFMaskUNet(nn.Module):
    """U-Net source separation model in the time-frequency domain.

    Uses the standard `soft-masking` technique to separate a single
    constituent audio source from its input mixture. The architecture
    implements a vanilla U-Net design, which involves a basic
    encoder-decoder scheme (without an additional bottleneck layer).
    The separation procedure is as follows:

    * The encoder first compresses an audio sample x in the time-frequency
      domain to a low-resolution representation.
    * The decoder receives the encoder's output as input, and reconstructs
      a new sample x~, which matches the dimensionality of x.
    * A an activation layer normalizes x~ to force its values to be between
      [0, 1], creating a `soft-mask`.
    * The mask is applied to the original audio sample x as an element-wise
      product, yielding the target source estimate y.

    Args:
        num_fft_bins (int): Number of STFT bins (otherwise known as filter
            banks). Note that only num_fft_bins // 2 + 1 are used due to the
            symmetry property of the fourier transform.
        num_samples (int): Number of samples (temporal dimension).
        num_channels (int): Number of audio channels.
        max_depth (int): Maximum depth of the autoencoder. Default: 6.
        hidden_size (int): Initial hidden size of the autoencoder.
            Default: 16.
        kernel_size (union[Tuple, int]): Kernel sizes of the autoencoder. A
            tuple of (enc_conv_size, downsampler_size, upsampler_size,
            dec_conv_size) may be passed in. Otherwise, all kernels will
            share the same size. Default: 3.
        block_size (int): Depth of each encoder/decoder block. Default: 3.
        downsampler (str): Downsampling method employed by the encoder.
            Default: 'max_pool'.
        upsampler (str): Upsampling method employed by the decoder.
            Default: 'transpose".
        batch_norm (bool): Whether to use batch normalization. Default: True.
        layer_activation_fn (str): Activation function used for each
            autoencoder layer. Default: 'relu'.
        mask_activation_fn (str): Final activation used to construct the
            multiplicative soft-mask. Default: 'relu'.
        dropout_p (float): Dropout probability. If p > 0, dropout is used.
            Default: 0.
        use_skip (bool): Whether to concatenate skipped audio from the encoder
            to the decoder. Default: True.
        normalize_input (bool): Whether to normalize the input. Note, that
            this layer simply uses batch norm instead of actual audio whitening.
            Default: False.
    """

    def __init__(
        self,
        num_fft_bins: int,
        num_samples: int,
        num_channels: int,
        max_depth: int = 6,
        hidden_size: int = 16,
        kernel_size: Union[Tuple, int] = 3,
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        layer_activation_fn: str = "relu",
        mask_activation_fn: str = "relu",
        dropout_p: float = 0,
        use_skip: bool = True,
        normalize_input: bool = False,
    ):
        # Note that only 1 source will be estimated per mask model.
        self.num_targets = 1
        self.num_fft_bins = num_fft_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.max_depth = max_depth
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.layer_activation_fn = layer_activation_fn
        self.mask_activation_fn = mask_activation_fn
        self.dropout_p = dropout_p
        self.use_skip = use_skip
        self.normalize_input = normalize_input
        super(TFMaskUNet, self).__init__()

        if self.normalize_input:
            self.input_norm = nn.BatchNorm2d(self.num_fft_bins // 2 + 1)

        self.autoencoder = AutoEncoder2d(
            num_targets=1,
            num_bins=self.num_fft_bins // 2 + 1,
            num_samples=self.num_samples,
            num_channels=self.num_channels,
            max_depth=max_depth,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            block_size=block_size,
            downsampler=downsampler,
            upsampler=upsampler,
            batch_norm=batch_norm,
            activation=layer_activation_fn,
            dropout_p=dropout_p,
            use_skip=use_skip,
        )

        self.mask_activation = _get_activation(
            activation_fn=mask_activation_fn
        )

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        if self.normalize_input:
            data = self.input_norm(data.permute(0, 2, 3, 1))
            data = data.permute(0, 3, 1, 2)
        data = self.autoencoder(data)
        mask = self.mask_activation(data)
        return mask

        # for _ in range(num_models):
        #     self.models.append(
        #         TFMaskUNet(
        #             num_fft_bins=dataset_params["num_fft"],
        #             num_samples=num_samples,
        #             num_channels=dataset_params["num_channels"],
        #             max_depth=arch_params["max_depth"],
        #             hidden_size=arch_params["hidden_size"],
        #             kernel_size=arch_params["kernel_size"],
        #             block_size=arch_params["block_size"],
        #             downsampler=arch_params["downsampler"],
        #             upsampler=arch_params["upsampler"],
        #             batch_norm=arch_params["batch_norm"],
        #             layer_activation_fn=arch_params["layer_activation_fn"],
        #             mask_act_fn=arch_params["mask_act_fn"],
        #             dropout_p=arch_params["dropout_p"],
        #             use_skip=arch_params["use_skip"],
        #             normalize_input=arch_params["normalize_input"],
        #         )
        #     )
