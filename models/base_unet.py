import torch
import torch.nn as nn

import numpy as np
from typing import Optional, Union
from models.layers import StackedEncoderBlock, StackedDecoderBlock


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.input_norm = nn.BatchNorm2d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, 2, 2), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, 2, 2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 5, 2, 2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 512, 5, 2, 2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.deconv1 = nn.ConvTranspose2d(512, 256, 5, 2, 2)
        self.a1 = nn.Sequential(nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.ReLU())
        self.deconv2 = nn.ConvTranspose2d(512, 128, 5, 2, 2)
        self.a2 = nn.Sequential(nn.BatchNorm2d(128), nn.Dropout2d(0.5), nn.ReLU())
        self.deconv3 = nn.ConvTranspose2d(256, 64, 5, 2, 2)
        self.a3 = nn.Sequential(nn.BatchNorm2d(64), nn.Dropout2d(0.5), nn.ReLU())
        self.deconv4 = nn.ConvTranspose2d(128, 32, 5, 2, 2)
        self.a4 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU())
        self.deconv5 = nn.ConvTranspose2d(64, 16, 5, 2, 2)
        self.a5 = nn.Sequential(nn.BatchNorm2d(16), nn.ReLU())
        self.deconv6 = nn.ConvTranspose2d(32, 1, 5, 2, 2)

        self.final_conv = nn.Conv2d(1, 1, 1, 1, padding='same')
        self.output_norm = nn.BatchNorm2d(512)
        self.a6 = nn.Sigmoid()

    def forward(self, x):

        x = self.input_norm(x)

        x = x.permute(0, 3, 1, 2)
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)

        d1 = self.a1(self.deconv1(e6, output_size=e5.size()))
        d2 = self.a2(self.deconv2(torch.cat([d1, e5], dim=1), output_size=e4.size()))
        d3 = self.a3(self.deconv3(torch.cat([d2, e4], dim=1), output_size=e3.size()))
        d4 = self.a4(self.deconv4(torch.cat([d3, e3], dim=1), output_size=e2.size()))
        d5 = self.a5(self.deconv5(torch.cat([d4, e2], dim=1), output_size=e1.size()))
        d6 = self.a6(self.deconv6(torch.cat([d5, e1], dim=1), output_size=x.size()))

        d6 = self.final_conv(d6)
        out = d6.permute(0, 2, 3, 1)
        out = self.output_norm(out)
        out = self.a6(out)

        output = {
            'mask': out,
            'estimate': None
        }

        return output


class DynamicUNet(nn.Module):
    """
    Dynamic implementation of the deep U-Net encoder/decoder architecture for
    source separation proposed in [1].

    Parameters
    ----------
    init_feature_size (int): number of starting features (default = 16)
    num_fft (int): number of STFT frequency bins (default = 512)
    num_channels (int): number of input channels
    max_layers (int): max depth of the autoencoder layers; will infer the limit
                      on number of layers if optional bottleneck_size is specified
    bottleneck_size (int or None): number of features for bottleneck layer;
                                   if unspecified, will default to num_fft
    dropout (float): dropout probability p if dropout > 0 else no dropout used
                     (default = 0.5)
    activation (str): final activation function (default = 'sigmoid')
    # target_sources (int or list): number of target sources to learn; by default,
    #                               model will learn mask estimation for vocals
    #                               only (default = 1)

    References
    ----------

    [1] Jansson, A., Humphrey, E., Montecchio, N. , Bittner, R.,
    Kumar, A. & Weyde, T.  (2017). Singing voice separation with deep U-Net
    convolutional networks. Paper presented at the 18th International Society
    for Music Information Retrieval Conference, 23-27 Oct 2017, Suzhou, China.
    """
    def __init__(
            self,
            init_features: int = 16,
            num_fft: int = 512,
            num_channels: int = 1,
            bottleneck_size: Optional[int] = 512,
            bottleneck_layers: int = 1,
            bottleneck_type: str = 'conv',
            max_layers: int = 6,
            encoder_block_layers: int = 1,
            decoder_block_layers: int = 1,
            encoder_kernel_size: int = 5,
            decoder_kernel_size: int = 5,
            dropout: float = 0.5,
            dropout_layers: int = 3,
            skip_connections: bool = True,
            mask_activation: str = 'sigmoid',
            block_size: int = 2,
            downsampling_method: Optional[str] = 'conv',
            upsampling_method: Optional[str] = 'transposed',
            block_activation: str = 'relu',
            leakiness: Optional[float] = 0.2,
            target_sources: Union[int, list] = 1,
            input_norm: bool = False,
            output_norm: bool = False
    ):
        super(DynamicUNet, self).__init__()

        assert num_fft & (num_fft - 1) == 0, \
            f"Frequency dimension must be a power of 2, but received {num_fft}"
        assert init_features & (init_features - 1) == 0, \
            f"Input feature size must be a power of 2, but received {num_fft}"

        self.init_feature_size = init_features
        self.num_channels = num_channels

        # determine depth of the model
        self.max_layers = min(
            max_layers,
            int(np.log2(num_fft // init_features) + 1)
        )

        self.bottleneck_size = bottleneck_size
        self.dropout = dropout

        # construct autoencoder layers
        encoder = []
        decoder = []
        for layer in range(self.max_layers):

            if layer == 0:
                in_channels = num_channels
                out_channels = init_features
            else:
                in_channels = out_channels
                out_channels = in_channels * 2

            encoder.append(
                StackedEncoderBlock(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    # downsampling_method=downsampling_method,
                    leak=leakiness,
                    layers=encoder_block_layers
                )
            )

            out_channels = out_channels // 2 if layer == self.max_layers - 1 else out_channels

            decoder.append(
                StackedDecoderBlock(
                    out_channels * 2,
                    in_channels,
                    kernel_size=5,
                    layers=decoder_block_layers,
                    dropout=0.5 if 0 < self.max_layers - layer <= 3 else 0,
                    skip_block=True
                )
            )

        # register layers and final activation
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.activation = nn.Sigmoid() if mask_activation == 'sigmoid' else nn.ReLU()

        # input normalization
        # self.input_normalization = nn.BatchNorm2d(num_fft)

    def forward(self, data: torch.Tensor) -> dict:
        """
        Performs source separation by indirectly learning a soft mask, and
        applying it to the mixture STFT (Short Time Fourier Transform).

        Parameters
        ----------
        data (tensor): mixture magnitude STFT data of shape (N, F, T, C), where
                       N: number of samples in mini-batch
                       F: number of frequency bins
                       T: number of frames (along the temporal dimension)
                       C: number of input channels (1 = mono, 2 = stereo)

        Returns
        -------
        output (tensor): source estimate matching the shape of input
        """

        # # make copy of mixture STFT
        # original = data.detach().clone().permute(0, 3, 1, 2)

        # normalize input
        # data = self.input_normalization(data)

        # channels must be first non-batch dimension for convolutional layers
        data = data.permute(0, 3, 1, 2)
        original = data

        encodings = []


        # downsamplng layers
        for layer in range(self.max_layers - 1):
            data = self.encoder[layer](data)
            encodings.append(data)
            # print(f"E{layer + 1}", data.mean())
            # save non-bottleneck intermediate encodings for skip connection

        # pass through bottleneck layer
        data = self.encoder[-1](data)
        data = self.decoder[-1](data, encodings[-1].size())

        # upsampling layers
        for layer in range(self.max_layers - 2):
            # print(f"D{layer + 1}", self.decoder[-1 - layer].skip_block)
            data = torch.cat([data, encodings[-1 - layer]], dim=1)
            data = self.decoder[-2 - layer](data, encodings[-2 - layer].size())
            # print(f"D{layer + 1}", data.mean())

        # final conv layer + activation
        data = torch.cat([data, encodings[0]], dim=1)
        data = self.decoder[0](data, original.size())
        mask = self.activation(data)

        # print(mask.mean())

        # get source estimate
        # output = original.clone().detach() * mask

        # reshape to match input shape
        # output = output.permute(0, 2, 3, 1)

        # stash copy of mask for signal reconstruction
        mask = mask.permute(0, 2, 3, 1)

        output = {
            'estimate': None,
            'mask': mask
        }

        return output

    def separate(self, stft):
        with torch.no_grad():
            mag, phase = torch.abs(stft), torch.angle(stft)

            mask = self.forward(mag)['mask']

            estimate = (mask * mag) * torch.exp(1j * phase)
        return estimate

