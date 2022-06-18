# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import math
import torch
import numpy as np

from librosa import amplitude_to_db
from typing import Optional, Tuple, Callable, List, Union
from torch import Tensor
from torchaudio import transforms
from numpy import ndarray


class AudioTransform(object):
    """Wrapper class that conveniently stores multiple transformation tools.

    Args:
        num_fft (int): Number of FFT bins.
        hop_length (int): Hop length.
        window_size (int): Window size.
        sample_rate (int): Sample rate. Default; 44100.
        device (str): Device. Default: 'cpu'.
    """

    def __init__(
        self,
        num_fft: int,
        hop_length: int,
        window_size: int,
        sample_rate: int = 44100,
        device: str = "cpu",
    ):
        super(AudioTransform, self).__init__()
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.window_size = window_size

        self.stft = transforms.Spectrogram(
            n_fft=num_fft,
            win_length=window_size,
            hop_length=hop_length,
            power=None,
            onesided=True,
        )

        self.inv_stft = transforms.InverseSpectrogram(
            n_fft=num_fft,
            win_length=window_size,
            hop_length=hop_length,
            onesided=True,
        )

        self.mel_scale = transforms.MelScale(
            n_mels=64,
            f_min=1,
            f_max=16384,
            sample_rate=sample_rate,
            n_stft=num_fft // 2 + 1,
            # norm="slaney",
        )

        # Use librosa implementation due to discrepancy w/ torchaudio.
        self.amp_to_db_ = amplitude_to_db

        # Transfer window functions and filterbanks to GPU if available.
        self.stft.window = self.stft.window.to(device)
        self.inv_stft.window = self.inv_stft.window.to(device)
        self.mel_scale.fb = self.mel_scale.fb.to(device)

    def to_decibel(self, spectrogram: Tensor) -> Tensor:
        r"""Transforms spectrogram to decibel scale.

        Computes the following:
        .. math::
            y = 20 \mul \\log_{10} \\frac{|x|}{max(|x|)}

        Args:
            spectrogram (Tensor): Spectrogram data.

        Returns:
            spectrogram (Tensor): Log-normalized spectrogram data.
        """
        log_normal = torch.from_numpy(
            self.amp_to_db_(spectrogram.cpu(), ref=np.max)
        ).to(spectrogram.device)
        return log_normal

    def to_spectrogram(
        self, audio: Tensor, use_padding: bool = True
    ) -> Tensor:
        """Transforms an audio signal to its time-frequency representation.

        Args:
            audio (Tensor): Raw audio signal data.
            use_padding (bool): If ``True``, zero-pads audio data such that its
                duration is a multiple of the hop length. Default: True.

        Returns:
            Tensor: Spectrogram data (complex-valued).
        """
        if use_padding:
            audio = self.pad_audio(audio)
        return self.stft(audio)

    def to_audio(self, complex_spec: Tensor) -> Tensor:
        """Transforms complex-valued spectrogram to its time-domain signal.

        Args:
            complex_spec (Tensor): Complex spectrogram data.

        Returns:
            Tensor: Audio signal data.
        """
        return self.inv_stft(complex_spec)

    def to_mel_scale(self, spectrogram: Tensor, to_db: bool = True) -> Tensor:
        """Transforms a magnitude spectrogram to a mel spectrogram.

        Args:
            spectrogram (Tensor): Magnitude spectrogram data.
            to_db (bool): If ``True``, converts magnitude spectrogram to
                decibel scale first. Default: True.

        Returns:
            Tensor: Mel spectrogram data.
        """
        mel_spectrogram = self.mel_scale(spectrogram)
        if to_db:
            mel_spectrogram = self.to_decibel(mel_spectrogram)
        return mel_spectrogram

    def audio_to_mel(self, audio: Tensor, to_db: bool = True) -> Tensor:
        """Transforms raw audio signal to a mel spectrogram.

        Uses ``to_spectrogram`` and ``to_mel_scale`` as subroutines.

        Args:
            audio (Tensor): Raw audio signal data.
            to_db (bool): If ``True``, converts magnitude spectrogram to
                decibel scale first. Default: True.

        Returns:
            Tensor: Mel spectrogram data.
        """
        spectrogram = self.to_spectrogram(audio)
        amp_spectrogram = torch.abs(spectrogram)
        mel_spectrogram = self.to_mel_scale(amp_spectrogram, to_db=to_db)
        return mel_spectrogram

    def pad_audio(self, audio: Tensor):
        """Zero-pads audio to make its duration divisible by the hop length.

        Args:
            audio (Tensor): Audio signal data.

        Returns:
            Tenor: Zero-padded audio signal data.
        """
        remainder = int(audio.shape[-1] % self.hop_length)
        pad_size = self.hop_length - remainder
        padding = torch.zeros(
            size=(*audio.shape[:-1], pad_size), device=audio.device
        )
        audio = torch.cat([audio, padding], dim=-1)
        return audio


def fast_fourier(transform: Callable, audio: Tensor) -> Tensor:
    """Transforms raw audio to complex-valued STFT audio."""
    audio_stft = []
    n_batch, n_channels, n_frames, n_targets = audio.size()

    for i in range(n_channels):
        sources_stack = []
        for j in range(n_targets):
            sources_stack.append(
                transform(audio[:, i, :, j].reshape((n_batch, n_frames)))
            )
        audio_stft.append(torch.stack(sources_stack, dim=-1))

    data_stft = torch.stack(audio_stft, dim=1)
    return data_stft


def inverse_fast_fourier(transform: Callable, complex_stft: Tensor):
    """Transforms complex-valued STFT audio to temporal audio domain."""
    source_estimate = []
    n_batch, n_channels, n_frames, n_targets = complex_stft.size()

    for i in range(n_targets):
        source_estimate.append(transform(complex_stft[:, :, :, i].squeeze(-1)))

    source_estimate = torch.stack(source_estimate, dim=-1)
    return source_estimate


def get_stft(
    num_fft: int,
    hop_length: int,
    window_size: int,
    inverse: bool = False,
    use_hann: bool = True,
    trainable: bool = False,
    device: str = "cpu",
) -> Callable:
    """Returns a specified stft or inverse stft transform."""

    if use_hann:
        window = make_hann_window(window_size, trainable, device)
    else:
        window = None

    stft_fn = torch.stft if not inverse else torch.istft

    def transform(data):
        stft_data = stft_fn(
            input=data,
            n_fft=num_fft,
            hop_length=hop_length,
            window=window,
            win_length=window_size,
            onesided=True,
            return_complex=True if not inverse else False,
        )
        return stft_data

    return transform


def get_num_stft_frames(
    sample_len: int, sr: int, win_size: int, hop_len: int, center: bool = True
) -> int:
    """Calculates number of STFT frames."""
    # Force number of samples to be divisible by the hop length.
    n_samples = sample_len * sr
    remainder = n_samples % hop_len
    n_samples += hop_len - remainder

    win_size = (1 - bool(center)) * win_size
    frames = math.floor((n_samples - win_size) / hop_len + 1)
    return frames


def make_hann_window(
    window_length: int, trainable: bool = False, device: Optional[str] = None
) -> torch.Tensor:
    """Creates a ``Hann`` window for use with STFT/ISTFT transformations.

    Args:
        window_length (int): Window length.
        trainable (bool): True if the window should be learned rather than
            fixed. Default: False.
        device (optional[str]): Device.

    Returns:
        (Tensor): Window as a tensor.
    """
    window = torch.hann_window(
        window_length=window_length,
        device=device if device is not None else "cpu",
        requires_grad=trainable,
    )
    return window


def get_deconv_pad(
    h_in: int, w_in: int, h_out: int, w_out: int, stride: int, kernel_size: int
) -> Tuple[int, int]:
    """Computes the required transpose conv padding for a target shape."""
    h_pad = math.ceil((kernel_size - h_out + stride * (h_in - 1)) / 2)
    w_pad = math.ceil((kernel_size - w_out + stride * (w_in - 1)) / 2)
    assert h_pad >= 0 and w_pad >= 0
    return h_pad, w_pad


def get_conv_pad(
    h_in: int, w_in: int, h_out: int, w_out: int, kernel_size: int
) -> Tuple[int, int]:
    """Computes the required conv padding."""
    h_pad = max(0, math.ceil((2 * h_out - 2 + kernel_size - h_in) / 2))
    w_pad = max(0, math.ceil((2 * w_out - 2 + kernel_size - w_in) / 2))
    assert h_pad >= 0 and w_pad >= 0
    return h_pad, w_pad


def get_conv_shape(
    h_in: int, w_in: int, stride: int, kernel_size: int
) -> Tuple[int, int]:
    """Computes the non-zero padded output of a conv layer."""
    h_out = math.floor((h_in - kernel_size) / stride + 1)
    w_out = math.floor((w_in - kernel_size) / stride + 1)
    assert h_out >= 0 and w_out >= 0
    return h_out, w_out


def trim_audio(
    audio_tensors: Union[List[Tensor], List[ndarray]]
) -> Union[List[Tensor], List[ndarray]]:
    """Trims duration of all audio data to match that of the shortest one.

    Args:
        audio_tensors (Union[List[Tensor], List[ndarray]]): List of audio
            tensors or arrays.
    Returns:
        Union[List[Tensor], List[ndarray]]: Trimmed audio tensors or arrays.
    """
    if len(audio_tensors) and isinstance(audio_tensors[0], np.ndarray):
        assert all(
            [
                len(aud.shape) == len(audio_tensors[0].shape)
                for aud in audio_tensors
            ]
        )
        if len(audio_tensors[0].shape) == 2:
            audio_tensors = [
                np.expand_dims(aud, axis=0) for aud in audio_tensors
            ]
    elif len(audio_tensors):
        assert all(
            [aud.dim() == audio_tensors[0].dim() for aud in audio_tensors]
        )
        if audio_tensors[0].dim() == 2:
            audio_tensors = [aud.unsqueeze(0) for aud in audio_tensors]
    n_frames = min(audio_tensors, key=lambda aud: aud.shape[-1]).shape[-1]
    trimmed_audio = [aud[:, :, :n_frames] for aud in audio_tensors]
    return trimmed_audio
