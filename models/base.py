import inspect

import torch
import torch.nn as nn
import numpy as np

from typing import List, Union, Optional, Tuple, Any
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from transforms import STFT, InverseSTFT
from .modules import AutoEncoder2d, VAE2d
from .layers import _get_activation
from transforms import _make_hann_window
from functools import wraps
from inspect import Signature
from dataclasses import dataclass

# __all__ = ['UNetTFMaskEstimate', 'UNetTFSourceEstimate']


# def separate(
#         self, signal: torch.FloatTensor, power_spectrum: bool = False
# ) -> torch.FloatTensor:
#     """"""
#     mixture_data = self.stft(signal)
#     mixture_mag_data = torch.abs(mixture_data)
#     mixture_phase_data = torch.angle(mixture_data)
#     source_mask = self.forward(mixture_mag_data)
#     estimate_mag_data = source_mask * mixture_mag_data
#     estimate_phase_corrected = estimate_mag_data * torch.exp(
#         1j * mixture_phase_data
#     )
#     estimate_signal = self.istft(estimate_phase_corrected)
#
#     return estimate_signal

class SeparationModel(nn.Module):
    """Interface for all base source separation models.

    Not meant to be implemented directly, but subclassed instead.
    All source separation models implement forward, backward, separate
    and inference methods.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config: dict):
        super(SeparationModel, self).__init__()
        self.is_training = config['mode']

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, mixture_data, target_data):
        pass

    @abstractmethod
    def update_params(self):
        pass

    @abstractmethod
    def separate(self, audio):
        pass

    @abstractmethod
    def inference(self, audio):
        pass

    @abstractmethod
    def validate(self):
        pass









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

