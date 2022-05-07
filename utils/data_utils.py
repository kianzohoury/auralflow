import torch
import math

from typing import Optional, Tuple, Callable
from torch import stft, istft


def get_stft(
    num_fft: int,
    hop_length: int,
    window_size: int,
    inverse: bool = False,
    use_hann: bool = True,
    trainable: bool = False,
    device: str = 'cuda'
) -> Callable:
    """Returns a specified stft or inverse stft transform."""

    if use_hann:
        window = make_hann_window(window_size, trainable, device)
    else:
        window = None

    stft_fn = torch.stft if inverse else torch.istft

    def transform(data):
        stft_data = stft_fn(
            input=data,
            n_fft=num_fft,
            hop_length=hop_length,
            window=window,
            win_length=window_size,
            onesided=True,
            return_complex=True,
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
