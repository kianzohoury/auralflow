import torch.nn as nn


class DynamicUNet(nn.Module):
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
            window_size: int = 1024,
            hop_length: int = 768,
            num_channels: int = 1,
            targets: Optional[List] = None,
            bottleneck_layers: int = 3,
            bottleneck_type: str = 'conv',
            max_layers: int = 6,
            encoder_block_layers: int = 1,
            decoder_block_layers: int = 1,
            encoder_kernel_size: int = 5,
            decoder_kernel_size: int = 5,
            encoder_down: Optional[str] = 'conv',
            decoder_up: Optional[str] = 'transposed',
            decoder_dropout: float = 0.5,
            num_dropouts: int = 3,
            skip_connections: bool = True,
            encoder_leak: float = 0.2,
            mask_activation: str = 'sigmoid',
            input_norm: bool = False,
            output_norm: bool = False
    ):
        super(DynamicUNet, self).__init__()
        assert num_fft & (num_fft - 1) == 0, \
            f"Frequency dimension must be a power of 2, but received {num_fft}"
        assert init_features & (init_features - 1) == 0, \
            f"Input feature size must be a power of 2, but received {num_fft}"

        self.init_feature_size = init_features
        self.num_fft = num_fft
        self.window_size = window_size
        self.hop_length = hop_length
        self.targets = targets if targets else ['vocals']
        self.bottleneck_layers = bottleneck_layers
        self.bottleneck_type = bottleneck_type
        self.num_channels = num_channels

        # determine depth of the model
        self.max_layers = min(
            max_layers,
            int(np.log2(num_fft // init_features + 1e-6) + 1)
        )

        # construct autoencoder layers
        encoder = nn.ModuleList()
        decoder = nn.ModuleList()
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
                    layers=encoder_block_layers,
                    kernel_size=encoder_kernel_size,
                    downsampling_method=encoder_down,
                    leak=encoder_leak
                )
            )

            if layer == self.max_layers - 1:
                out_channels = out_channels // 2

            decoder.append(
                StackedDecoderBlock(
                    out_channels * 2,
                    in_channels,
                    kernel_size=decoder_kernel_size,
                    layers=decoder_block_layers,
                    dropout=decoder_dropout if 0 < self.max_layers - layer <= num_dropouts else 0,
                    skip_block=True
                )
            )

        mid_channels = encoder[-1].out_channels
        bottleneck = nn.ModuleList()
        if bottleneck_type == 'conv':
            for _ in range(bottleneck_layers):
                bottleneck.append(
                    nn.Sequential(
                        nn.Conv2d(mid_channels, mid_channels, encoder_kernel_size, 1, 'same'),
                        nn.BatchNorm2d(mid_channels),
                        nn.ReLU()
                    )
                )
        elif bottleneck_type == 'lstm':
            pass
        else:
            pass

        self.bottleneck = bottleneck


        # Final 1x1 conv layer for multi-source mask estimation.
        final_num_channels = decoder[0].out_channels

        # if len(self.targets) > 1:
        self.final_conv = nn.Conv2d(
            in_channels=final_num_channels,
            out_channels=len(self.targets),
            kernel_size=1,
            stride=1,
            padding='same'
        )
        # else:
        # self.final_conv = nn.Identity()

        # register layers and final activation
        self.encoder = encoder
        self.decoder = decoder
        self.activation = nn.Sigmoid() if mask_activation == 'sigmoid' else nn.ReLU()
        self.input_norm = nn.BatchNorm2d(512)
        # self.output_norm = nn.BatchNorm2d(num_fft // 2)

        # if input_norm:
        #     self.input_norm = nn.BatchNorm2d(num_fft)
        # else:
        #     self.input_norm = nn.Identity()
        # if output_norm:
        #     self.output_norm = nn.BatchNorm2d(num_fft)
        # else:
        #     self.output_norm = nn.Identity()

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


        # normalize input
        # data = self.input_normalization(data)

        # channels must be first non-batch dimension for convolutional layers
        data = self.input_norm(data)
        data = data.permute(0, 3, 1, 2)
        original = data

        encodings = []

        # downsamplng layers
        for layer in range(len(self.encoder)):
            data = self.encoder[layer](data)
            if layer < len(self.encoder) - 1:
                encodings.append(data)

        for i in range(len(self.bottleneck)):
            data = self.bottleneck[i](data)

        # upsampling layers
        for layer in range(len(self.decoder)):
            if layer == 0:
                output_size = encodings[-1].size()
                data = self.decoder[-1](data, output_size)
            elif layer < len(self.encoder) - 1:
                output_size = encodings[-1 - layer].size()
                data = torch.cat([data, encodings[-layer]], dim=1)
                data = self.decoder[-1 - layer](data, output_size)
            else:
                output_size = original.size()
                data = torch.cat([data, encodings[-layer]], dim=1)
                data = self.decoder[-1 - layer](data, output_size)
        # final conv layer + activation
        data = self.final_conv(data)
        mask = self.activation(data)

        # print(mask.mean())

        # get source estimate
        # output = original.clone().detach() * mask

        # reshape to match input shape
        # output = output.permute(0, 2, 3, 1)

        # stash copy of mask for signal reconstruction
        mask = mask.permute(0, 2, 3, 1)
        # mask = self.output_norm(mask)

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


