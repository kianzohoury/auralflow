import torch
import torch.nn as nn

import numpy as np
from typing import Optional, Union, List
from models.layers import StackedEncoderBlock, StackedDecoderBlock


class BaseUNet(nn.Module):
    """
    Dynamic implementation of the deep U-Net encoder/decoder architecture for
    source separation proposed in [1].

    Args:
    ----------
    init_features (int):
        Number of starting features. Default: 16.
    num_fft (int):
        Number of STFT frequency bins. Default: 512.
    num_channels (int):
        Number of input channels. Default: 1.
    max_layers (int):
        max depth of the autoencoder layers; will infer the limit n number of
        layers if optional bottleneck_size is specified.
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

        num_targets: int = 1,

        max_layers: int = 6,
        bottleneck_layers: int = 0,
        bottleneck_type: str = 'conv',
        # residual: bool = False,

        encoder_scheme: Optional[dict] = None,
        decoder_scheme: Optional[dict] = None,

        # encoder_block_layers: int = 1,
        # decoder_block_layers: int = 1,
        # encoder_kernel_size: Union[int, List] = 5,
        # decoder_kernel_size: Union[int, List] = 5,
        # encoder_down: Optional[str] = 'conv',
        decoder_up: Optional[str] = 'transposed',
        decoder_dropout: float = 0.5,
        num_dropouts: int = 3,
        skip_connections: bool = True,
        # encoder_leak: float = 0.2,
        final_conv_layer=False,


        mask_activation: str = 'sigmoid',
        input_norm: bool = False,
        output_norm: bool = False
    ):
        super(BaseUNet, self).__init__()

        assert num_fft & (num_fft - 1) == 0, \
            f"Frequency dimension must be a power of 2, but received {num_fft}"
        assert init_features & (init_features - 1) == 0, \
            f"Input feature size must be a power of 2, but received {init_features}"
        assert 0 < num_targets <= 4,\
            f"Number of targets must be between 0 and 4, but received {num_targets}"
        assert 1 < num_channels < 2, \
            f"Number of channels must be either 1 (mono) or 2 (stereo), but received {num_channels}"
        assert bottleneck_type in {'conv', 'lstm', 'linear'}, \
            f"Bottleneck type must be one of the following: 'conv', 'lstm', 'linear'."

        # Register model attributes.
        self.init_feature_size = init_features
        self.num_fft = num_fft
        self.num_targets = num_targets
        self.bottleneck_layers = bottleneck_layers
        self.bottleneck_type = bottleneck_type
        self.num_channels = num_channels
        self.skip_connections = skip_connections
        self.final_conv_layer = final_conv_layer
        # self.residual = residual

        # Correct the maximum depth of the model if needed.
        self.max_layers = min(
            max_layers,
            int(np.log2(num_fft // init_features + 1e-6) + 1)
        )

        # Build the autoencoder.
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        in_channels, out_channels = num_channels, init_features

        for layer in range(self.max_layers):
            if layer == 0:
                in_channels = num_channels
                out_channels = init_features
            else:
                in_channels = out_channels
                out_channels = in_channels * 2

            self.encoder.append(
                StackedEncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scheme=encoder_scheme
                )
            )

            if layer == self.max_layers - 1:
                out_channels = out_channels // 2

            if self.max_layers - layer <= num_dropouts:
                dropout_p = decoder_dropout
            else:
                dropout_p = 0

            self.decoder.append(
                StackedDecoderBlock(
                    in_channels=out_channels * (1 + int(skip_connections)),
                    out_channels=in_channels,
                    scheme=decoder_scheme,
                    dropout_p=dropout_p,
                    skip_block=skip_connections
                )
            )

        # Build the bottleneck.
        mid_channels = self.encoder[-1].out_channels

        if bottleneck_type == 'conv':
            self.bottleneck = nn.ModuleList()
            for _ in range(bottleneck_layers):
                self.bottleneck.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding='same'
                    ),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU())
                )
        elif bottleneck_type == 'lstm':
            self.bottleneck = nn.Identity()
        elif bottleneck_type == 'linear':
            self.bottleneck = nn.Identity()
        else:
            self.bottleneck = nn.Identity()

        # Build final 1x1 conv layer.
        final_num_channels = self.decoder[0].out_channels

        if self.final_conv_layer:
            self.soft_conv = nn.Conv2d(
                in_channels=final_num_channels,
                out_channels=num_targets,
                kernel_size=1,
                stride=1,
                padding='same'
            )
        else:
            self.soft_conv = nn.Identity()

        if mask_activation == 'sigmoid':
            self.mask_activation = nn.Sigmoid()
        else:
            self.mask_activation = nn.ReLU()

        if input_norm:
            self.input_norm = nn.BatchNorm2d(num_fft)
        else:
            self.input_norm = nn.Identity()

        if output_norm:
            self.output_norm = nn.BatchNorm2d(num_fft)
        else:
            self.output_norm = nn.Identity()

    def forward(self, data: torch.Tensor) -> dict:
        """
        Performs source separation by indirectly learning a soft mask, and
        applying it to the mixture STFT (Short Time Fourier Transform).

        Args:
        data (tensor):
            mixture magnitude STFT data of shape (N, F, T, C), where
            N: number of samples in mini-batch
            F: number of frequency bins
            T: number of frames (along the temporal dimension)
            C: number of input channels (1 = mono, 2 = stereo)

        Returns:
            output (tensor): source estimate matching the shape of input
        """

        data = self.input_norm(data)

        # Switch audio channel to the first non-batch dimension.
        data = data.permute(0, 3, 1, 2)
        # Store the input shape for later.
        input_size = data.shape

        # Store intermediate feature maps if utilizing skip connections.
        encodings = []

        # Feed into the encoder layers.
        for layer in range(len(self.encoder)):
            data = self.encoder[layer](data)
            if layer < len(self.encoder) - 1 and self.skip_connections:
                encodings.append(data)

        # Pass through bottleneck.
        data = self.bottleneck(data)

        # Feed into the decoder layers.
        for layer in range(len(self.decoder)):
            if layer < len(self.decoder) - 1:
                output_size = encodings[-1 - layer].size()
                if layer > 0 and self.skip_connections:
                    data = torch.cat([data, encodings[-layer]], dim=1)
                data = self.decoder[-1 - layer](data, output_size)
            else:
                if self.skip_connections:
                    data = torch.cat([data, encodings[-layer]], dim=1)
                data = self.decoder[-1 - layer](data, input_size)

        # Final conv + output normalization + mask activation.
        data = self.soft_conv(data)
        data = self.output_norm(data)
        mask = self.activation(data)

        # Reshape to match the input size.
        mask = mask.permute(0, 2, 3, 1)

        return mask
