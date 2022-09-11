# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import importlib
import torch


from auralflow.transforms import AudioTransform, _get_num_stft_frames
from .base import SeparationModel
from torch import autocast, FloatTensor, Tensor
from typing import Dict, List, Tuple


class SpectrogramMaskModel(SeparationModel):
    """Spectrogram-domain deep mask estimation model.

    Wraps a ``torch.nn.Module`` model.

    :ivar audio_transform: Data transformer/processer.
    :vartype audio_transform: AudioTransform

    Args:
        model_type (str): Base model architecture.
        targets (List[str]): Target source labels.
        num_fft (int): Number of FFT bins. Note that only
            ``num_fft`` // 2 + 1 bins will be used due to symmetry.
            Default: ``1024``.
        window_size (int): Window size. Default: ``1024``.
        hop_length (int): Hop length. Default: ``512``.
        sample_length (int): Duration of audio samples in seconds.
            Default: ``3``.
        sample_rate (int): Sample rate. Default: ``44100``.
        num_channels (int): Number of input/output channels.
            Default: ``1``.
        num_hidden_channels (int): Number of initial hidden channels.
            Default: ``16``.
        mask_act_fn (str): Mask activation function.
            Default: ``'sigmoid'``.
        leak_factor (float): If ``leak_factor`` > 0, network will
            have Leaky ReLU activation layers, and SELU activation layers
            otherwise. Default: ``0``.
        dropout_p (float): Layer dropout probability.
            Default: ``0.4``.
        normalize_input (bool): Whether to learn input
            scaling/centering parameters. Default: ``True``.
        normalize_output (bool): Whether to learn output
            scaling/centering parameters. Default: ``True``.
        device (str): Device. Default: ``'cpu'``.

    Keyword Args:
        kwargs: Additional constructor arguments (depending on the
            ``base_model_type``).
    """

    audio_transform: AudioTransform

    def __init__(
        self,
        model_type: str,
        targets: List[str],
        num_fft: int = 1024,
        window_size: int = 1024,
        hop_length: int = 512,
        sample_length: int = 3,
        sample_rate: int = 44100,
        num_channels: int = 1,
        num_hidden_channels: int = 16,
        mask_act_fn: str = "sigmoid",
        leak_factor: float = 0,
        dropout_p: float = 0.4,
        normalize_input: bool = True,
        normalize_output: bool = True,
        device: str = 'cpu',
        **kwargs
    ):
        super(SpectrogramMaskModel, self).__init__()

        self._targets = sorted(targets)
        self._device = device

        # Calculate number of frames (temporal dimension).
        self.num_stft_frames = _get_num_stft_frames(
            sample_len=sample_length,
            sr=sample_rate,
            win_size=window_size,
            hop_len=hop_length,
            center=True,
        )

        # Note that the num bins will be num_fft // 2 + 1 due to symmetry.
        self.n_fft_bins = num_fft // 2 + 1
        self.num_out_channels = num_channels
        self._multi_estimator = len(self.targets) > 1

        # Retrieve requested base model architecture class.
        base_class = getattr(
            importlib.import_module("auralflow.models"),
            model_type
        )

        # Create the model instance and set to current device.
        self._model = base_class(
            num_fft_bins=self.n_fft_bins,
            num_frames=self.num_stft_frames,
            num_channels=self.num_out_channels,
            hidden_channels=num_hidden_channels,
            mask_act_fn=mask_act_fn,
            leak_factor=leak_factor,
            dropout_p=dropout_p,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            device=self.device,
            **kwargs
        ).to(self.device)

        # Instantiate data transformer for pre/post audio processing.
        self.audio_transform = AudioTransform(
            num_fft=num_fft,
            hop_length=hop_length,
            window_size=window_size,
            device=self.device
        )

    def to_spectrogram(self, audio: Tensor) -> Tuple[FloatTensor, FloatTensor]:
        """Transforms audio signal data to spectrogram data.

        Given a raw audio signal, a complex spectrogram is produced and then
        split into its constituent magnitude and phase content. Transfers
        ``audio`` to the same device as the model.

        Args:
            audio (Tensor): Audio signal of dimension
                `(batch, channels, time)`.

        Returns:
            Tuple[FloatTensor, FloatTensor]: Magnitude and phase
            spectrograms, respectively, both of dimension
            `(batch, channels, freq, frames)`.
        """
        # Compute complex-valued STFT and send tensor to GPU if available.
        # Squeeze the last dimension (target sources) if possible.
        complex_stft = self.audio_transform.to_spectrogram(
            audio.squeeze(-1).to(self.device)
        )
        # Separate magnitude and phase.
        mag_spec = torch.abs(complex_stft).float()
        phase_spec = torch.angle(complex_stft).float()
        return mag_spec, phase_spec

    def to_audio(
        self, estimate: FloatTensor, phase: FloatTensor
    ) -> FloatTensor:
        """Transforms the target estimate from spectrogram data to audio data.

        Uses the "dirty phase" approximation trick, which utilizes the phase
        content of the mixture spectrogram to reconstruct the audio signal of
        the estimated target spectrogram.

        Args:
            estimate (FloatTensor): Estimated target source spectrogram
                of dimension `(batch, channels, freq, frames)`.
            phase (FloatTensor): Phase spectrogram of dimension
                `(batch, channels, freq, frames)`.

        Returns:
            FloatTensor: Estimated target source audio signal of dimension
            `(batch, channels, time)`.
        """
        phase_corrected = estimate * torch.exp(1j * phase)
        estimate_audio = self.audio_transform.to_audio(phase_corrected).float()
        return estimate_audio

    def forward(
        self, mixture: FloatTensor
    ) -> Tuple[FloatTensor, Dict[str, FloatTensor]]:
        r"""Estimates the magnitude spectrogram of the target source.

        Applies the learned soft-mask (output of the network) via an
        element-wise product, to the magnitude spectrogram of the mixture,
        extracting an estimate of the target source. Given the magnitude
        spectrogram of a mixture, :math:`|X|`, it returns:

        .. math::

            \hat Y = M_{\theta} \odot |X|

        where :math:`M_{\theta}` is the learned soft-mask and :math:`\hat Y`
        is the target source estimate as a magnitude spectrogram.

        Args:
            mixture (torch.FloatTensor): Magnitude spectrogram of the mixture,
                of dimension `(batch, channels, freq, frames)`.

        Returns:
            Tuple[FloatTensor, Dict[str, FloatTensor]]: Magnitude spectrogram
            of the estimated target source, of dimension
            `(batch, channels, freq, frames)`, followed by a dictionary mapping
            ``'mask'`` to the soft-mask estimate, of dimension
            `(batch, channels, freq, frames)`, ``'mu'`` and ``'sigma'``
            to the distribution parameters: :math:`\mu, \sigma`, respectively,
            if ``self.model`` is an instance of ``SpectrogramNetVAE``.
        """
        output = self.model(mixture)
        # Handle network output.
        if isinstance(output, tuple):
            mask, mu, sigma = output
            data = {"mask": mask, "mu": mu, "sigma": sigma}
        else:
            mask = output
            data = {"mask": mask}
        estimate_spec = mask * (mixture.clone().detach())
        return estimate_spec, data

    def separate(self, mixture: Tensor) -> Tensor:
        r"""Extracts the target source audio signal given a mixture.

        Runs ``forward(...)`` in evaluation mode, and handles pre/post audio
        transformations.

        .. math::

            \hat S = \text{iSTFT}(\hat Y \odot \text{exp}(j \odot
                \angle_{\phi} X))

        where :math:`\text{iSTFT}` is the inverse short-time Fourier transform,
        :math:`\hat Y` is the target source estimate as a magnitude
        spectrogram, :math:`j` is imaginary and :math:`\angle_{\phi} X` is
        the element-wise angle of the `complex` spectrogram of the mixture.

        Args:
            mixture (Tensor): Mixture audio signal of dimension
                `(batch, channels, time)`

        Returns:
            Tensor: Target source audio signal of dimension
            `(batch, channels, time)`.
        """
        self.eval()
        with torch.no_grad():
            mixture_mag, mixture_phase = self.to_spectrogram(audio=mixture)
            estimate_spec, mask = self.forward(mixture=mixture_mag)
            estimate_audio = self.to_audio(
                estimate=estimate_spec, phase=mixture_phase
            )
        return estimate_audio
