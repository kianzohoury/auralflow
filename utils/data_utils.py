import torch
import math
import librosa

from typing import Optional, Tuple, Callable
from torchaudio import transforms
from torch import Tensor
import numpy as np



class AudioTransform(object):
    """Wrapper class that conveniently shares parameters between transforms."""
    def __init__(
        self,
        num_fft: int,
        hop_length: int,
        window_size: int,
        sample_rate: int = 44100,
        device: str = 'cpu'
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
            onesided=True
        )

        self.mel_scale = transforms.MelScale(
            n_mels=256,
            sample_rate=sample_rate,
            n_stft=num_fft // 2 + 1,
            norm="slaney"
        )

        # Transfer window functions and filterbanks to GPU if available.
        self.stft.window = self.stft.window.to(device)
        self.inv_stft.window = self.inv_stft.window.to(device)
        self.mel_scale.fb = self.mel_scale.fb.to(device)

    def to_spectrogram(self, audio: Tensor) -> Tensor:
        """Transforms an audio signal to its time-freq representation."""
        return self.stft(audio)

    def to_audio(self, complex_spec: Tensor) -> Tensor:
        """Transforms complex-valued spectrogram to its time-domain signal."""
        return self.inv_stft(complex_spec)

    def to_decibel(self, spectrogram: Tensor) -> Tensor:
        """Transforms spectrogram to decibel scale."""
        # Use implementation from librosa due to discrepancy w/ torchaudio.
        log_normal = torch.from_numpy(
            librosa.amplitude_to_db(spectrogram, ref=np.max)
        ).to(spectrogram.device)
        return log_normal

    def to_mel_scale(self, spectrogram: Tensor, to_db: bool = False) -> Tensor:
        if to_db:
            spectrogram = self.to_decibel(spectrogram)
        return self.mel_scale(spectrogram)

    def audio_to_mel(self, audio: Tensor):
        spectrogram = self.to_spectrogram(audio)
        amp_spectrogram = torch.abs(spectrogram)
        dec_spectrogram = self.to_decibel(amp_spectrogram)
        mel_spectrogram = self.to_mel_scale(dec_spectrogram)
        return mel_spectrogram


doc_str = """The
detailed procedure is as follows:

* X = |STFT(A)|
* S = X * M
* S_p = X * M * P
* A_s = iSTFT(S_p)

* where X: magnitude spectrogram;
* S: output of forward pass
* S_p: phase corrected output S
* P: complex-valued phase matrix, i.e., exp^(i * theta), where theta
is the angle between the real and imaginary parts of STFT(A).
* A_s: estimate source signal converted from time-freq to time-only
domain."""


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
        source_estimate.append(
            transform(complex_stft[:, :, :, i].squeeze(-1))
        )

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


def get_num_frames(
        sample_rate: int,
        sample_length: int,
        num_fft: int,
        window_size: int,
        hop_length: int,
) -> int:
    """Returns the number of FFT/STFT frequency bins."""
    x = torch.rand((1, sample_rate * sample_length))
    y = torch.stft(
        x,
        n_fft=num_fft,
        win_length=window_size,
        hop_length=hop_length,
        onesided=True,
        return_complex=True,
    )
    return y.size(-1)


def make_hann_window(
        window_length: int, trainable: bool = False, device: Optional[str] = None
) -> torch.Tensor:
    """Creates a `Hann` window for use with STFT/ISTFT transformations.

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
