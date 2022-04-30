import torch
import torch.nn as nn

from typing import List, Optional


def _make_hann_window(
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
        dtype=torch.float,
        device=device if device is not None else "cpu",
        requires_grad=trainable,
    )
    return window


class STFT(nn.Module):
    """Wrapper class for PyTorch's functional stft method.

    Implements a multi-channel, multi-target way of getting the time-frequency
    representation of a mini-batch of audio data.
    (`Short Time Fourier Transform`).

    Args:
        num_targets (int): Number of target sources.
        num_channels (int): Number of audio channels.
        num_fft (int): Number of fourier bins.
        hop_length (int): Hop length.
        window_length (int): Window length.
        window_type (optional[str]):
        trainable (bool): True if the window should be learned rather than
            fixed. Default: False.
    """

    def __init__(
        self,
        num_targets: int,
        num_channels: int,
        num_fft: int,
        hop_length: int,
        window_length: int,
        window_type: Optional[str] = None,
        trainable: bool = False,
    ):
        super(STFT, self).__init__()
        if num_fft < hop_length:
            raise ValueError(
                "Number of FFT bins must be at least as large as the hop"
                f"length {hop_length}, but received {num_fft}."
            )
        self.num_targets = num_targets
        self.num_channels = num_channels
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.win_length = window_length
        if window_type == "hann_window":
            self.window = _make_hann_window(
                window_length=window_length,
                trainable=trainable,
                device=self.device,
            )
        else:
            self.window = None

    def forward(self, audio: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method. Applies the `STFT` transform to audio data.

        Args:
            audio (Tensor): A batch of raw audio

        Returns:
            (Tensor): STFT audio.

        """

        complex_stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            onesided=True,
            return_complex=True,
        )
        return complex_stft


# output = []
# for target in range(random_data.shape[-1]):
#     channel_stfts = []
#     for channel in range(random_data.shape[1]):
#         stft_output = torch.stft(
#             random_data[:, channel, :, target].squeeze(1).squeeze(-1),
#             n_fft=num_fft - 1,
#             hop_length=hop_length,
#             win_length=window_size - 1,
#             onesided=True,
#             return_complex=True,
#         )
#         channel_stfts.append(stft_output)
#     output.append(torch.stack(channel_stfts, dim=-1))
# output = torch.stack(output, dim=-1)


class InverseSTFT(nn.Module):
    def __init__(
        self, n_fft, hop_length, win_length, window_type, trainable=False
    ):
        super(InverseSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if window_type == "hann_window":
            self.window = make_hann_window(
                window_length=win_length, trainable=trainable, device=device
            )
        else:
            self.window = None

    def forward(self, input_stft):
        audio_signal = torch.istft(
            input_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            onesided=True,
        )

        return audio_signal


# class PhaseApproximation(nn.Module):
#     """
#     Implements a common phase approximation trick. Utilizes phase
#     content from mixture STFT to approximate a phase-corrected (but noisy)
#     source STFT.
#     """
#     def __init__(
#             self,
#             phase: torch.Tensor
#     ):
#         super(PhaseApproximation, self).__init__()
#         self.phase = phase
#
#     def forward(self, source: torch.Tensor) -> torch.Tensor:
#         return source * torch.exp(1j * self.phase)


def get_data_shape(
    batch_size: int,
    sample_rate: int,
    sample_length: int,
    num_fft: int,
    window_size: int,
    hop_length: int,
    num_channels: int,
    targets: List,
):
    """Helper method to get the shape of PyTorch's STFT implementation.

    Returns:
        (tensor.Size): Size of the output.
    """

    # Due to symmetry property of STFT, if we want F frequency bins,
    # we need to specify (F - 1) n_ffts, since num_fft = F // 2 + 1.
    random_data = torch.rand(
        (batch_size, num_channels, sample_rate * sample_length, len(targets))
    )
    output = []
    for target in range(random_data.shape[-1]):
        channel_stfts = []
        for channel in range(random_data.shape[1]):
            stft_output = torch.stft(
                random_data[:, channel, :, target].squeeze(1).squeeze(-1),
                n_fft=num_fft - 1,
                hop_length=hop_length,
                win_length=window_size - 1,
                onesided=True,
                return_complex=True,
            )
            channel_stfts.append(stft_output)
        output.append(torch.stack(channel_stfts, dim=-1))
    output = torch.stack(output, dim=-1)
    return output.size()
