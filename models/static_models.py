import torch
import torch.nn as nn

from utils.data_utils import get_deconv_pad


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, leak=0.2):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leak),
        )

    def forward(self, data):
        return self.conv(data)


class UNetBlockDouble(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, leak=0.2):
        super(UNetBlockDouble, self).__init__()
        self.conv = nn.Sequential(
            UNetBlock(in_channels, out_channels, kernel_size, leak),
            UNetBlock(out_channels, out_channels, kernel_size, leak),
        )

    def forward(self, data):
        return self.conv(data)


class UNetBlockTriple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, leak=0.2):
        super(UNetBlockTriple, self).__init__()
        self.conv = nn.Sequential(
            UNetBlock(in_channels, out_channels, kernel_size, leak),
            UNetBlock(out_channels, out_channels, kernel_size, leak),
            UNetBlock(out_channels, out_channels, kernel_size, leak),
        )

    def forward(self, data):
        return self.conv(data)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size=3,
        kernel_size=3,
        leak=0.2,
        compress=True,
    ):
        super(DownBlock, self).__init__()
        if size == 3:
            self.conv_block = UNetBlockTriple(
                in_channels, out_channels, kernel_size, leak
            )
        elif size == 2:
            self.conv_block = UNetBlockDouble(
                in_channels, out_channels, kernel_size, leak
            )
        else:
            self.conv_block = UNetBlock(
                in_channels, out_channels, kernel_size, leak
            )
        if compress:
            self.down = nn.MaxPool2d(2)
        else:
            self.down = nn.Identity()

    def forward(self, data):
        skip = self.conv_block(data)
        output = self.down(skip)
        return output, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        transpose_padding,
        size=3,
        kernel_size=3,
    ):
        super(UpBlock, self).__init__()
        if size == 3:
            self.conv_block = UNetBlockTriple(
                in_channels, out_channels, kernel_size, leak=0
            )
        elif size == 2:
            self.conv_block = UNetBlockDouble(
                in_channels, out_channels, kernel_size, leak=0
            )
        else:
            self.conv_block = UNetBlock(
                in_channels, out_channels, kernel_size, leak=0
            )
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=2,
            padding=transpose_padding,
        )

    def forward(self, data, skip_data):
        data = self.up(data, output_size=skip_data.size())
        data = self.conv_block(torch.cat([data, skip_data], dim=1))
        return data


class SpectrogramNetSimple(nn.Module):
    def __init__(
        self,
        num_fft_bins,
        num_samples,
        num_channels=1,
        block_size=3,
        hidden_dim=16,
        mask_activation_fn="sigmoid",
        leak_factor=0.2,
    ):
        super(SpectrogramNetSimple, self).__init__()

        self.down_1 = DownBlock(
            num_channels, hidden_dim, size=block_size, leak=leak_factor
        )
        self.down_2 = DownBlock(
            hidden_dim, hidden_dim * 2, size=block_size, leak=leak_factor
        )
        self.down_3 = DownBlock(
            hidden_dim * 2, hidden_dim * 4, size=block_size, leak=leak_factor
        )
        self.down_4 = DownBlock(
            hidden_dim * 4, hidden_dim * 8, size=block_size, leak=leak_factor
        )
        self.down_5 = DownBlock(
            hidden_dim * 8, hidden_dim * 16, size=block_size, leak=leak_factor
        )
        self.down_6 = DownBlock(
            hidden_dim * 16,
            hidden_dim * 32,
            size=block_size,
            leak=leak_factor,
            compress=False,
        )

        enc_sizes = [[num_fft_bins >> l, num_samples >> l] for l in range(6)]

        self.up_1 = UpBlock(
            hidden_dim * 32,
            hidden_dim * 16,
            transpose_padding=get_deconv_pad(
                *enc_sizes[-1], *enc_sizes[-2], stride=2, kernel_size=5
            ),
            size=block_size,
        )

        self.up_2 = UpBlock(
            hidden_dim * 16,
            hidden_dim * 8,
            transpose_padding=get_deconv_pad(
                *enc_sizes[-2], *enc_sizes[-3], stride=2, kernel_size=5
            ),
            size=block_size,
        )
        self.up_3 = UpBlock(
            hidden_dim * 8,
            hidden_dim * 4,
            transpose_padding=get_deconv_pad(
                *enc_sizes[-3], *enc_sizes[-4], stride=2, kernel_size=5
            ),
            size=block_size,
        )
        self.up_4 = UpBlock(
            hidden_dim * 4,
            hidden_dim * 2,
            transpose_padding=get_deconv_pad(
                *enc_sizes[-4], *enc_sizes[-5], stride=2, kernel_size=5
            ),
            size=block_size,
        )
        self.up_5 = UpBlock(
            hidden_dim * 2,
            hidden_dim,
            transpose_padding=get_deconv_pad(
                *enc_sizes[-5],
                num_fft_bins,
                num_samples,
                stride=2,
                kernel_size=5
            ),
            size=block_size,
        )
        self.soft_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.mask_activation = (
            nn.Sigmoid() if mask_activation_fn == "sigmoid" else nn.ReLU()
        )

    def forward(self, data):
        data = data.permute(0, 3, 1, 2)
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
        mask = mask.permute(0, 2, 3, 1).unsqueeze(-1)
        return mask
