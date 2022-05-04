from typing import Tuple

import torch
import torch.nn as nn

from utils.data_utils import get_deconv_pad
from torch import FloatTensor
from losses import KLDivergenceLoss
from torch.nn import L1Loss


class ConvBlock(nn.Module):
    """Conv => Batch Norm => ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, leak=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leak, inplace=True),
        )

    def forward(self, data):
        output = self.conv(data)
        return output


class ConvBlockTriple(nn.Module):
    """(Conv => Batch Norm => ReLU) x 3 block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, leak=0.2):
        super(ConvBlockTriple, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, leak),
            ConvBlock(out_channels, out_channels, kernel_size, leak),
            ConvBlock(out_channels, out_channels, kernel_size, leak),
        )

    def forward(self, data):
        output = self.conv(data)
        return output


class DownBlock(nn.Module):
    """Downsampling convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        leak: float = 0.2,
        reduce: bool = True,
    ):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlockTriple(
            in_channels, out_channels, kernel_size, leak
        )
        if reduce:
            self.down = nn.MaxPool2d(kernel_size=2)
        else:
            self.down = nn.Identity()

    def forward(self, data: FloatTensor) -> Tuple[FloatTensor, ...]:
        skip = self.conv_block(data)
        output = self.down(skip)
        return output, skip


class UpBlock(nn.Module):
    """Upsampling convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: Tuple[int, int],
        kernel_size: int = 3,
    ):
        super(UpBlock, self).__init__()
        self.conv_block = ConvBlockTriple(
            in_channels, out_channels, kernel_size, leak=0
        )
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=2,
            padding=padding,
        )

    def forward(self, data: FloatTensor, skip: FloatTensor) -> FloatTensor:
        data = self.up(data, output_size=skip.size())
        data = self.conv_block(torch.cat([data, skip], dim=1))
        return data


class SpectrogramNetSimple(nn.Module):
    """Vanilla spectrogram U-Net model with triple block sizes."""
    criterion: L1Loss

    def __init__(
        self,
        num_fft_bins: int,
        num_samples: int,
        num_channels: int = 1,
        hidden_dim: int = 16,
        mask_activation_fn: str = "sigmoid",
        leak_factor: float = 0.2,
    ):
        super(SpectrogramNetSimple, self).__init__()
        self.criterion = L1Loss()
        self.channel_sizes = [[num_channels, hidden_dim]]
        self.channel_sizes += [
            [hidden_dim << l, hidden_dim << (l + 1)] for l in range(5)
        ]

        self.down_1 = DownBlock(*self.channel_sizes[0], leak=leak_factor)
        self.down_2 = DownBlock(*self.channel_sizes[1], leak=leak_factor)
        self.down_3 = DownBlock(*self.channel_sizes[2], leak=leak_factor)
        self.down_4 = DownBlock(*self.channel_sizes[3], leak=leak_factor)
        self.down_5 = DownBlock(*self.channel_sizes[4], leak=leak_factor)
        self.down_6 = DownBlock(
            *self.channel_sizes[5], leak=leak_factor, reduce=False
        )

        self.encoding_sizes = [
            [num_fft_bins >> l, num_samples >> l] for l in range(6)
        ]
        padding_sizes = [
            get_deconv_pad(
                *self.encoding_sizes[-1 - l],
                *self.encoding_sizes[-2 - l],
                stride=2,
                kernel_size=5
            )
            for l in range(len(self.encoding_sizes) - 1)
        ]

        self.channel_sizes = [size[::-1] for size in self.channel_sizes]

        self.up_1 = UpBlock(*self.channel_sizes[-1], padding=padding_sizes[0])
        self.up_2 = UpBlock(*self.channel_sizes[-2], padding=padding_sizes[1])
        self.up_3 = UpBlock(*self.channel_sizes[-3], padding=padding_sizes[2])
        self.up_4 = UpBlock(*self.channel_sizes[-4], padding=padding_sizes[3])
        self.up_5 = UpBlock(*self.channel_sizes[-5], padding=padding_sizes[4])

        self.channel_sizes = [size[::-1] for size in self.channel_sizes]

        self.soft_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        if mask_activation_fn == "sigmoid":
            self.mask_activation = nn.Sigmoid()
        elif mask_activation_fn == "relu":
            self.mask_activation = nn.ReLU()
        else:
            self.mask_activation = nn.Identity()

    def forward(self, data: FloatTensor) -> FloatTensor:
        enc_1, skip_1 = self.down_1(data)
        enc_2, skip_2 = self.down_2(enc_1)
        enc_3, skip_3 = self.down_3(enc_2)
        enc_4, skip_4 = self.down_4(enc_3)
        enc_5, skip_5 = self.down_5(enc_4)
        enc_6, _ = self.down_6(enc_5)

        dec_1 = self.up_1(enc_6, skip_5)
        dec_2 = self.up_2(dec_1, skip_4)
        dec_3 = self.up_3(dec_2, skip_3)
        dec_4 = self.up_4(dec_3, skip_2)
        dec_5 = self.up_5(dec_4, skip_1)
        dec_6 = self.soft_conv(dec_5)

        mask = self.mask_activation(dec_6)
        return mask


class SpectrogramLSTM(SpectrogramNetSimple):
    """Spectrogram U-Net with an LSTM bottleneck."""

    def __init__(
        self, *args, lstm_layers: int = 3, lstm_hidden_size=512, **kwargs
    ):
        super(SpectrogramLSTM, self).__init__(*args, **kwargs)
        num_features = self.channel_sizes[-1][-1] * self.encoding_sizes[-1][-1]
        self.num_features = num_features

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            num_layers=lstm_layers,
        )

        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden_size, num_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        enc_1, skip_1 = self.down_1(data)
        enc_2, skip_2 = self.down_2(enc_1)
        enc_3, skip_3 = self.down_3(enc_2)
        enc_4, skip_4 = self.down_4(enc_3)
        enc_5, skip_5 = self.down_5(enc_4)
        enc_6, _ = self.down_6(enc_5)

        n, c, b, t = enc_6.size()
        enc_6 = enc_6.permute(0, 2, 1, 3).reshape((n, b, c * t))

        lstm_out, _ = self.lstm(enc_6)
        lstm_out = lstm_out.reshape((n * b, -1))

        latent_data = self.linear(lstm_out)
        latent_data = latent_data.reshape((n, b, c, t)).permute(0, 2, 1, 3)

        dec_1 = self.up_1(latent_data, skip_5)
        dec_2 = self.up_2(dec_1, skip_4)
        dec_3 = self.up_3(dec_2, skip_3)
        dec_4 = self.up_4(dec_3, skip_2)
        dec_5 = self.up_5(dec_4, skip_1)
        dec_6 = self.soft_conv(dec_5)

        mask = self.mask_activation(dec_6)
        return mask


class SpectrogramLSTMVariational(SpectrogramLSTM):
    """Spectrogram U-Net model with a variational autoencoder + LSTM."""

    latent_data: FloatTensor
    mu_data: FloatTensor
    sigma_data: FloatTensor
    criterion: KLDivergenceLoss

    def __init__(self, *args, **kwargs):
        super(SpectrogramLSTMVariational, self).__init__(*args, **kwargs)
        self.criterion = KLDivergenceLoss()
        self.mu = nn.Linear(self.num_features, self.num_features)
        self.sigma = nn.Linear(self.num_features, self.num_features)
        self.eps = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.eps.loc = self.eps.loc.cuda()
            self.eps.scale = self.eps.scale.cuda()

    def forward(self, data: FloatTensor):
        enc_1, skip_1 = self.down_1(data)
        enc_2, skip_2 = self.down_2(enc_1)
        enc_3, skip_3 = self.down_3(enc_2)
        enc_4, skip_4 = self.down_4(enc_3)
        enc_5, skip_5 = self.down_5(enc_4)
        enc_6, _ = self.down_6(enc_5)

        n, c, b, t = enc_6.shape

        enc_6 = enc_6.permute(0, 2, 1, 3).reshape((n, b, c * t))

        self.mu_data = self.mu(enc_6)
        self.sigma_data = torch.exp(self.sigma(enc_6)).float()
        eps = self.eps.sample(sample_shape=self.sigma_data.shape)
        self.latent_data = self.mu_data + self.sigma_data * eps

        lstm_out, _ = self.lstm(self.latent_data)
        lstm_out = lstm_out.reshape((n * b, -1))

        dec_0 = self.linear(lstm_out)
        dec_0 = dec_0.reshape((n, b, -1, t)).permute(0, 2, 1, 3)

        dec_1 = self.up_1(dec_0, skip_5)
        dec_2 = self.up_2(dec_1, skip_4)
        dec_3 = self.up_3(dec_2, skip_3)
        dec_4 = self.up_4(dec_3, skip_2)
        dec_5 = self.up_5(dec_4, skip_1)
        dec_6 = self.soft_conv(dec_5)

        mask = self.mask_activation(dec_6)
        return mask
