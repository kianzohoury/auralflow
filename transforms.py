import torch
import torch.nn as nn

from typing import List


def make_hann_window(window_length, trainable=True, device=None):
    window = torch.hann_window(
        window_length=window_length,
        dtype=torch.float,
        device=device if device is not None else 'cpu',
        requires_grad=trainable
    )
    return window


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window_type, trainable=False):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if window_type == "hann_window":
            self.window = make_hann_window(
                window_length=win_length,
                trainable=trainable,
                device=device
            )
        else:
            self.window = None

    def forward(self, audio):
        complex_stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            onesided=True,
            return_complex=True
        )

        return complex_stft


class InverseSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window_type, trainable=False):
        super(InverseSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if window_type == "hann_window":
            self.window = make_hann_window(
                window_length=win_length,
                trainable=trainable,
                device=device
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


class PhaseApproximation(nn.Module):
    """
    Implements a common phase approximation trick. Utilizes phase
    content from mixture STFT to approximate a phase-corrected (but noisy)
    source STFT.
    """
    def __init__(
            self,
            phase: torch.Tensor
    ):
        super(PhaseApproximation, self).__init__()
        self.phase = phase

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * torch.exp(1j * self.phase)


def get_data_shape(batch_size: int, sample_rate: int, sample_length: int,
                   num_fft: int, window_size: int, hop_length: int,
                   num_channels: int, targets: List):
    """Helper method to get the shape of PyTorch's STFT implementation.

    Returns:
        (tensor.Size): Size of the output.
    """

    # Due to symmetry property of STFT, if we want F frequency bins,
    # we need to specify (F - 1) n_ffts, since num_fft = F // 2 + 1.
    random_data = torch.rand((batch_size, num_channels,
                              sample_rate * sample_length, len(targets)))
    output = []
    for target in range(random_data.shape[-1]):
        channel_stfts = []
        for channel in range(random_data.shape[1]):
            stft_output = torch.stft(
                random_data[:, channel, :, target].squeeze(1).squeeze(-1),
                n_fft=num_fft - 1, hop_length=hop_length,
                win_length=window_size - 1,
                onesided=True, return_complex=True)
            channel_stfts.append(stft_output)
        output.append(torch.stack(channel_stfts, dim=-1))
    output = torch.stack(output, dim=-1)
    return output.size()
