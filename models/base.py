import torch
import torch.nn as nn
import numpy as np

from typing import List, Union, Optional, Tuple
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from transforms import STFT, InverseSTFT
from .modules import AutoEncoder2d, VAE2d
from .layers import _get_activation


class SeparationModel(nn.Module):
    __metaclass__ = ABCMeta
    """Interface for all base source separation models."""

    def __init__(self):
        super(SeparationModel, self).__init__()

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, mixture_data, target_data):
        pass

    @abstractmethod
    def separate(self, audio):
        pass

    @abstractmethod
    def inference(self, audio):
        pass


class TFMaskModelBase(nn.Module):
    """Base class for deep source mask estimation in the time-frequency domain.

    Args:


    Returns:

    """

    def __init__(
        self,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        num_fft: int,
        hop_length: int,
        window_size: int,
        num_targets: int,
    ):
        super(TFMaskModelBase, self).__init__()

        self.num_bins = num_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.num_targets = num_targets

    # def backward(
    #     self, mixture_data: torch.FloatTensor, target_data: torch.FloatTensor
    # ) -> float:
    #     """"""
    #     loss = self._loss_fn(mixture_data, target_data)
    #     loss_val = loss.item()
    #     loss.backward()
    #     return loss_val

    # def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
    #     return data

    def separate(
        self, signal: torch.FloatTensor, power_spectrum: bool = False
    ) -> torch.FloatTensor:
        """"""
        mixture_data = self.stft(signal)
        mixture_mag_data = torch.abs(mixture_data)
        mixture_phase_data = torch.angle(mixture_data)
        source_mask = self.forward(mixture_mag_data)
        estimate_mag_data = source_mask * mixture_mag_data
        estimate_phase_corrected = estimate_mag_data * torch.exp(
            1j * mixture_phase_data
        )
        estimate_signal = self.istft(estimate_phase_corrected)

        return estimate_signal

    # def inference(self, audio):
    #     """"""
    #     return audio


class UNetTF(TFMaskModelBase):
    """U-Net source separation model in the time-frequency domain."""
    def __init__(
        self,
        num_targets: int,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        num_fft: int,
        hop_length: int,
        window_size: int,
        max_depth: int,
        hidden_size: int,
        kernel_size: Union[Tuple, int],
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        layer_activation: str = "relu",
        dropout_p: float = 0,
        use_skip: bool = True,
        normalize_input: bool = False,
        mask_activation: str = "relu",
    ):
        super(UNetTF, self).__init__(
            num_bins=num_bins,
            num_samples=num_samples,
            num_channels=num_channels,
            num_fft=num_fft,
            hop_length=hop_length,
            window_size=window_size,
            num_targets=num_targets,
        )

        self.max_depth = max_depth
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.layer_activation = layer_activation
        self.dropout_p = dropout_p
        self.use_skip = use_skip
        self.normalize_input = normalize_input
        self.mask_activation = mask_activation

        if self.normalize_input:
            self.input_norm = nn.BatchNorm2d(num_bins)

        self.autoencoder = AutoEncoder2d(
            num_targets=num_targets,
            num_bins=num_bins,
            num_samples=num_samples,
            num_channels=num_channels,
            max_depth=max_depth,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            block_size=block_size,
            downsampler=downsampler,
            upsampler=upsampler,
            batch_norm=batch_norm,
            activation=layer_activation,
            dropout_p=dropout_p,
            use_skip=use_skip,
        )
        self.mask_activation = _get_activation(activation_fn=mask_activation)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        data = self.input_norm(data) if self.normalize_input else data
        data = data.permute(0, 3, 1, 2)
        data = self.autoencoder.forward(data)
        mask = self.mask_activation(data)
        mask = mask.permute(0, 2, 3, 1, 4)
        return mask


class UNetVTF(TFMaskModelBase):
    """U-Net source separation model in the time-frequency domain."""
    def __init__(
        self,
        num_targets: int,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        num_fft: int,
        hop_length: int,
        window_size: int,
        max_depth: int,
        hidden_size: int,
        latent_size: int,
        kernel_size: Union[Tuple, int],
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        layer_activation: str = "relu",
        dropout_p: float = 0,
        use_skip: bool = True,
        normalize_input: bool = False,
        mask_activation: str = "relu",
    ):
        super(UNetVTF, self).__init__(
            num_bins=num_bins,
            num_samples=num_samples,
            num_channels=num_channels,
            num_fft=num_fft,
            hop_length=hop_length,
            window_size=window_size,
            num_targets=num_targets,
        )

        self.max_depth = max_depth
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.layer_activation = layer_activation
        self.dropout_p = dropout_p
        self.use_skip = use_skip
        self.normalize_input = normalize_input
        self.mask_activation = mask_activation

        if self.normalize_input:
            self.input_norm = nn.BatchNorm2d(num_bins)

        self.autoencoder = VAE2d(
            latent_size=latent_size,
            num_targets=num_targets,
            num_bins=num_bins,
            num_samples=num_samples,
            num_channels=num_channels,
            max_depth=max_depth,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            block_size=block_size,
            downsampler=downsampler,
            upsampler=upsampler,
            batch_norm=batch_norm,
            activation=layer_activation,
            dropout_p=dropout_p,
            use_skip=use_skip,
        )
        self.mask_activation = _get_activation(activation_fn=mask_activation)

    def forward(self, data: torch.FloatTensor) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        """Forward method."""
        data = self.input_norm(data) if self.normalize_input else data
        data = data.permute(0, 3, 1, 2)
        output, latent_dist = self.autoencoder.forward(data)
        mask = self.mask_activation(output)
        mask = mask.permute(0, 2, 3, 1, 4)
        return mask, latent_dist


# class MaskModel(TFMaskModelBase):
#     def __init__(self, **kwargs):
#         super(MaskModel, self).__init__(**kwargs)
#
#     # def forward(self, audio) -> torch.FloatTensor:
#     #     pass
#
#
# class AudioMaskModelBase(nn.Module):
#     """Base class for deep source mask estimation directly in the time domain."""
