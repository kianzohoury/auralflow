from typing import Tuple, Optional

import torch
import torch.nn as nn

from utils.data_utils import get_deconv_pad
from torch import FloatTensor, Tensor
from losses import losses
from torch.nn import L1Loss

torch.backends.cudnn.benchmark = True


class ConvBlock(nn.Module):
    """Conv => Batch Norm => ReLU block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bn=True,
        leak=0,
        dropout=0,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=not bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.LeakyReLU(leak)
        self.dropout = nn.Dropout2d(dropout, inplace=True)

    def forward(self, data):
        data = self.conv(data)
        data = self.relu(data)
        data = self.bn(data)
        output = self.dropout(data)
        return output


class ConvBlockTriple(nn.Module):
    """(Conv => Batch Norm => ReLU) x 3 block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, leak=0):
        super(ConvBlockTriple, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, False, leak, 0),
            ConvBlock(out_channels, out_channels, kernel_size, True, leak, 0)
            # ConvBlock(out_channels, out_channels, kernel_size, True, leak),
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
        leak: float = 0,
        reduce: bool = True,
    ) -> None:
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
    ) -> None:
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

    criterion: nn.Module
    residual_mask: FloatTensor

    def __init__(
        self,
        num_fft_bins: int,
        num_samples: int,
        num_channels: int = 1,
        hidden_dim: int = 16,
        mask_act_fn: str = "sigmoid",
        leak_factor: float = 0.2,
        normalize_input: bool = False,
    ) -> None:
        super(SpectrogramNetSimple, self).__init__()

        # Register attributes.
        self.num_fft_bins = num_fft_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.mask_activation_fn = mask_act_fn
        self.leak_factor = leak_factor
        self.normalize_input = normalize_input
        self.input_norm = nn.LayerNorm(
            (num_channels, num_fft_bins, num_samples)
        )

        # Use identity to prevent GPU from being slowed down in forward pass.
        # if normalize_input:
        #     self.input_norm = nn.BatchNorm2d(num_fft_bins)
        # else:
        #     self.input_norm = nn.Identity()

        # Calculate input/output channel sizes for each layer.
        self.channel_sizes = [[num_channels, hidden_dim]]
        self.channel_sizes += [
            [hidden_dim << i, hidden_dim << (i + 1)] for i in range(6)
        ]

        # Define encoder layers.
        self.down_1 = DownBlock(*self.channel_sizes[0], leak=leak_factor)
        self.down_2 = DownBlock(*self.channel_sizes[1], leak=leak_factor)
        self.down_3 = DownBlock(*self.channel_sizes[2], leak=leak_factor)
        self.down_4 = DownBlock(*self.channel_sizes[3], leak=leak_factor)
        self.down_5 = DownBlock(*self.channel_sizes[4], leak=leak_factor)
        self.down_6 = DownBlock(*self.channel_sizes[5], leak=leak_factor)

        # Define simple bottleneck layer.
        self.bottleneck = ConvBlockTriple(
            in_channels=self.channel_sizes[-1][0],
            out_channels=self.channel_sizes[-1][-1],
            leak=0,
        )

        # Determine the spatial dimension sizes for computing deconv padding.
        self.encoding_sizes = [
            [num_fft_bins >> i, num_samples >> i] for i in range(7)
        ]

        # Compute transpose/deconvolution padding.
        padding_sizes = [
            get_deconv_pad(
                *self.encoding_sizes[-1 - i],
                *self.encoding_sizes[-2 - i],
                stride=2,
                kernel_size=5
            )
            for i in range(len(self.encoding_sizes) - 1)
        ]

        dec_channel_sizes = [size[::-1] for size in self.channel_sizes][::-1]

        # Define decoder layers.
        self.up_1 = UpBlock(*dec_channel_sizes[0], padding=padding_sizes[0])
        self.up_2 = UpBlock(*dec_channel_sizes[1], padding=padding_sizes[1])
        self.up_3 = UpBlock(*dec_channel_sizes[2], padding=padding_sizes[2])
        self.up_4 = UpBlock(*dec_channel_sizes[3], padding=padding_sizes[3])
        self.up_5 = UpBlock(*dec_channel_sizes[4], padding=padding_sizes[4])
        self.up_6 = UpBlock(*dec_channel_sizes[5], padding=padding_sizes[5])

        # Final conv layer squeezes output channels dimension to num_channels.
        self.soft_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        # Define input/output normalization parameters.
        # self.input_center = nn.Parameter(
        #     torch.zeros(num_fft_bins).float(), requires_grad=True
        # )
        # self.input_scale = nn.Parameter(
        #     torch.ones(num_fft_bins).float(), requires_grad=True
        # )
        self.output_center = nn.Parameter(
            torch.zeros(num_fft_bins).float(), requires_grad=True
        )
        self.output_scale = nn.Parameter(
            torch.ones(num_fft_bins).float(), requires_grad=True
        )

        # Define activation function used for masking.
        if mask_act_fn == "sigmoid":
            self.mask_activation = nn.Sigmoid()
        elif mask_act_fn == "relu":
            self.mask_activation = nn.ReLU()
        elif mask_act_fn == "tanh":
            self.mask_activation = nn.Tanh()
        elif mask_act_fn == "softmask":
            self.mask_activation = nn.Softmax()
        elif mask_act_fn == "hardtanh":
            self.mask_activation = nn.Hardtanh(0, 1, inplace=True)
        elif mask_act_fn == "prelu":
            self.mask_activation = nn.PReLU()
        else:
            self.mask_activation = nn.Identity()

    def forward(self, data: FloatTensor) -> FloatTensor:
        """Forward method."""
        # Normalize input.
        data = self.input_norm(data)
        # data = self.input_norm(data.permute(0, 2, 3, 1))
        # data = data.permute(0, 3, 1, 2)
        # data = data.permute(0, 1, 3, 2)
        # data = data - self.input_center
        # data = data * self.input_scale
        # data = data.permute(0, 1, 3, 2)

        # Pass through encoder.
        enc_1, skip_1 = self.down_1(data)
        enc_2, skip_2 = self.down_2(enc_1)
        enc_3, skip_3 = self.down_3(enc_2)
        enc_4, skip_4 = self.down_4(enc_3)
        enc_5, skip_5 = self.down_5(enc_4)
        enc_6, skip_6 = self.down_6(enc_5)

        # Pass through bottleneck.
        latent_data = self.bottleneck(enc_6)

        # Pass through decoder.
        dec_1 = self.up_1(latent_data, skip_6)
        dec_2 = self.up_2(dec_1, skip_5)
        dec_3 = self.up_3(dec_2, skip_4)
        dec_4 = self.up_4(dec_3, skip_3)
        dec_5 = self.up_5(dec_4, skip_2)
        dec_6 = self.up_6(dec_5, skip_1)
        output = self.soft_conv(dec_6)

        # output = output.permute(0, 1, 3, 2)
        # output = output - self.output_center
        # output = output * self.output_scale
        # output = output.permute(0, 1, 3, 2)

        # Generate multiplicative soft-mask.
        mask = self.mask_activation(output)
        return mask


class SpectrogramLSTM(SpectrogramNetSimple):
    """Spectrogram U-Net with an LSTM bottleneck."""

    def __init__(
        self, *args, lstm_layers: int = 3, lstm_hidden_size=1024, **kwargs
    ) -> None:
        super(SpectrogramLSTM, self).__init__(*args, **kwargs)
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        # Calculate number of input features to the LSTM.
        n_features = self.channel_sizes[-1][0] * self.encoding_sizes[-1][-1]
        self.n_features = n_features

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            num_layers=lstm_layers,
            dropout=0.4
        )

        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden_size, n_features * 2),
            nn.ReLU(inplace=True)
            # nn.BatchNorm1d(n_features * 2),
        )

    def forward(self, data: FloatTensor) -> FloatTensor:
        """Forward method."""
        # Normalize input.
        data = self.input_norm(data)
        # data = self.input_norm(data.permute(0, 2, 3, 1))
        # data = data.permute(0, 3, 1, 2)
        # data = data.permute(0, 1, 3, 2)
        # data = data - self.input_center
        # data = data * self.input_scale
        # data = data.permute(0, 1, 3, 2)

        # Pass through encoder.
        enc_1, skip_1 = self.down_1(data)
        enc_2, skip_2 = self.down_2(enc_1)
        enc_3, skip_3 = self.down_3(enc_2)
        enc_4, skip_4 = self.down_4(enc_3)
        enc_5, skip_5 = self.down_5(enc_4)
        enc_6, skip_6 = self.down_6(enc_5)

        # Reshape encoded audio to pass through bottleneck.
        n, c, b, t = enc_6.size()
        enc_6 = enc_6.permute(0, 2, 1, 3).reshape((n, b, c * t))

        # Pass through recurrent stack.
        lstm_out, _ = self.lstm(enc_6)
        lstm_out = lstm_out.reshape((n * b, -1))

        # Project latent audio onto affine space and reshape for decoder.
        latent_data = self.linear(lstm_out)
        latent_data = latent_data.reshape((n, b, c * 2, t)).permute(0, 2, 1, 3)

        # Pass through decoder.
        dec_1 = self.up_1(latent_data, skip_6)
        dec_2 = self.up_2(dec_1, skip_5)
        dec_3 = self.up_3(dec_2, skip_4)
        dec_4 = self.up_4(dec_3, skip_3)
        dec_5 = self.up_5(dec_4, skip_2)
        dec_6 = self.up_6(dec_5, skip_1)
        output = self.soft_conv(dec_6)

        # Shift and scale output.
        # output = output.permute(0, 1, 3, 2)
        # output = output - self.output_center
        # output = output * self.output_scale
        # output = output.permute(0, 1, 3, 2)

        # Generate multiplicative soft-mask.
        mask = self.mask_activation(output)
        return mask


class SpectrogramLSTMVariational(SpectrogramLSTM):
    """Spectrogram U-Net model with a VAE and LSTM bottleneck.

    Encoder => VAE => LSTM x 3 => decoder. Models a Gaussian conditional
    distribution p(z|x) to sample latent variable z ~ p(z|x), to feed into
    decoder to generate x' ~ p(x|z).
    """

    latent_data: FloatTensor
    mu_data: FloatTensor
    sigma_data: FloatTensor

    def __init__(self, *args, **kwargs):
        super(SpectrogramLSTMVariational, self).__init__(*args, **kwargs)

        # Define normalizing flow layers.
        self.mu = nn.Linear(self.n_features, self.n_features)
        self.log_sigma = nn.Linear(self.n_features, self.n_features)
        self.eps = torch.distributions.Normal(0, 1)

        # Speed up sampling by utilizing GPU.
        if torch.cuda.is_available():
            self.eps.loc = self.eps.loc.cuda()
            self.eps.scale = self.eps.scale.cuda()

    def forward(self, data: FloatTensor) -> FloatTensor:
        """Forward method."""
        # Normalize input.

        # data = data.permute(0, 1, 3, 2)
        # data = data - self.input_center
        # data = data * self.input_scale
        # data = data.permute(0, 1, 3, 2)

        # data = self.input_norm(data.permute(0, 2, 3, 1))
        # data = data.permute(0, 3, 1, 2)

        # Pass through encoder.
        enc_1, skip_1 = self.down_1(data)
        enc_2, skip_2 = self.down_2(enc_1)
        enc_3, skip_3 = self.down_3(enc_2)
        enc_4, skip_4 = self.down_4(enc_3)
        # enc_5, skip_5 = self.down_5(enc_4)
        # enc_6, skip_6 = self.down_6(enc_5)

        # Reshape encodings to match dimensions of latent space.
        n, c, b, t = enc_4.shape
        enc_6 = enc_4.permute(0, 2, 1, 3).reshape((n, t, c * b))

        # Normalizing flow.
        self.mu_data = self.mu(enc_6)
        self.sigma_data = torch.exp(self.log_sigma(enc_6)).float()
        eps = self.eps.sample(sample_shape=self.sigma_data.shape)

        # Sample z from the modeled distribution.
        self.latent_data = self.mu_data + self.sigma_data * eps

        # Pass through recurrent stack.
        lstm_out, _ = self.lstm(self.latent_data)
        lstm_out = lstm_out.reshape((n * t, -1))

        # Pass through affine layers and reshape for decoder.
        dec_0 = self.linear(lstm_out)
        dec_0 = dec_0.reshape((n, b, -1, t)).permute(0, 2, 1, 3)

        # Pass through decoder.
        dec_1 = self.up_1(dec_0, skip_4)
        dec_2 = self.up_2(dec_1, skip_3)
        dec_3 = self.up_3(dec_2, skip_2)
        dec_4 = self.up_4(dec_3, skip_1)
        # dec_5 = self.up_5(dec_4, skip_2)
        # dec_6 = self.up_6(dec_5, skip_1)
        output = self.soft_conv(dec_4)

        output = output.permute(0, 1, 3, 2)
        output = output - self.output_center
        output = output * self.output_scale
        output = output.permute(0, 1, 3, 2)
        # Generate multiplicative soft-mask.
        mask = self.mask_activation(output)
        return mask

    def get_kl_div(self) -> Tensor:
        """Computes KL term."""
        return losses.kl_div_loss(self.mu_data, self.sigma_data)
