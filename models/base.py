
import torch
import torch.nn as nn
import numpy as np

from typing import List, Union, Optional
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from transforms import STFT, InverseSTFT


class SeparationModel(metaclass=ABCMeta):
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


class TFMaskModelBase(SeparationModel, nn.Module):
    """Base class for deep source mask estimation in the time-frequency domain.
    """
    def __init__(
        self,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        num_fft: int,
        hop_length: int,
        window_size: int,
        targets: Optional[List[str]],
        loss_fn: nn.Module,
    ):
        super(TFMaskModelBase, self).__init__()

        # Register model attributes.
        self._num_bins = num_bins
        self._num_samples = num_samples
        self._num_channels = num_channels
        self._num_fft = num_fft
        self._hop_length = hop_length
        self._window_size = window_size
        self._targets = targets if targets is not None else ['vocals']
        self._loss_fn = loss_fn
        self.stft = STFT(
            n_fft=num_fft,
            hop_length=hop_length,
            win_length=window_size,
            window_type='hann',
            trainable=False
        )
        self.istft = InverseSTFT(
            n_fft=num_fft,
            hop_length=hop_length,
            win_length=window_size,
            window_type='hann',
            trainable=False
        )

    def backward(
        self,
        mixture_data: torch.FloatTensor,
        target_data: torch.FloatTensor
    ) -> float:
        """"""
        loss = self._loss_fn(mixture_data, target_data)
        loss_val = loss.item()
        loss.backward()
        return loss_val

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return data

    def separate(
        self, signal: torch.FloatTensor,
        power_spectrum: bool = False
    ) -> torch.FloatTensor:
        """"""
        mixture_data = self.stft(signal)
        mixture_mag_data = torch.abs(mixture_data)
        mixture_phase_data = torch.angle(mixture_data)
        source_mask = self.forward(mixture_mag_data)
        estimate_mag_data = source_mask * mixture_mag_data
        estimate_phase_corrected = estimate_mag_data * torch.exp(1j * mixture_phase_data)
        estimate_signal = self.istft(estimate_phase_corrected)

        return estimate_signal

    def inference(self, audio):
        """"""
        return audio


class MaskModel(TFMaskModelBase):
    def __init__(self, **kwargs):
        super(MaskModel, self).__init__(**kwargs)

    # def forward(self, audio) -> torch.FloatTensor:
    #     pass


class AudioMaskModelBase(nn.Module):
    """Base class for deep source mask estimation directly in the time domain.

    """