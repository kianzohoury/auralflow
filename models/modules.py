from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import numpy as np


from models.layers import _AEBlock, _SoftConv, _get_conv_output_size
from typing import Tuple, Union, Optional, List

_MIN_BLOCK_SIZE = 1
_MAX_BLOCK_SIZE = 10
_MIN_AUTOENCODER_DEPTH = 2
_MAX_AUTOENCODER_DEPTH = 10
_TARGETS = ["bass", "drums", "vocals", "other"]
_BLOCK_TYPES = {"conv", "recurrent", "upsampler", "downsampler"}

__all__ = ["AutoEncoder2d", "VAE2d"]


class _AutoEncoder(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        num_targets: int,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        max_depth: int,
        hidden_size: int,
        kernel_size: Union[Tuple, int],
        same_padding: bool = True,
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        activation: str = "relu",
        leak_constant: Optional[float] = None,
        elu_constant: Optional[float] = None,
        prelu_n_params: Optional[int] = None,
        dropout_p: float = 0,
        use_skip: bool = True,
    ):
        super(_AutoEncoder, self).__init__()
        if num_targets < 0 or num_targets > 4:
            raise ValueError(
                "Number of targets must be between 1 and 4, but received"
                f" {num_targets}"
            )
        elif num_channels > 2 or num_channels < 1:
            raise ValueError(f"Channels must be 1 (mono) or 2 (stereo).")
        elif hidden_size < 0:
            raise ValueError(f"Hidden size must be at least 1.")
        elif isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 4 - bool(downsampler == "downsample"):
                raise ValueError(
                    "Must specify 4 kernel sizes, but received only"
                    f"{len(kernel_size)}."
                )
        elif block_size < _MIN_BLOCK_SIZE or block_size > _MAX_BLOCK_SIZE:
            raise ValueError(
                f"Block size must be between {_MIN_BLOCK_SIZE} and"
                f" {_MAX_BLOCK_SIZE}, but requested {block_size}"
            )

        self.num_targets = num_targets
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.same_padding = same_padding
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_p = dropout_p
        self.use_skip = use_skip

        # Put a ceiling on the depth of autoencoder.
        self.max_depth = max(
            min(max_depth, int(np.log2(num_bins // hidden_size + 1e-6) + 1)),
            _MIN_AUTOENCODER_DEPTH,
        )
        self.activation_param = leak_constant or elu_constant or prelu_n_params

        enc_conv_layers = nn.ModuleList()
        dec_conv_layers = nn.ModuleList()
        down_layers = nn.ModuleList()
        up_layers = nn.ModuleList()

        output_sizes = [[num_bins, num_samples]]
        in_channels, out_channels = num_channels, hidden_size

        for layer in range(self.max_depth):
            h_in, w_in = output_sizes[-1]
            enc_block_stack = []
            for _ in range(block_size):
                enc_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="conv",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding="same" if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=self.activation_param,
                        dropout_p=dropout_p,
                    )
                )
                if same_padding:
                    h_out, w_out = h_in, w_in
                else:
                    h_out, w_out = _get_conv_output_size(
                        h_in=h_in,
                        w_in=w_in,
                        stride=1,
                        kernel_size=kernel_size[0],
                    )
                output_sizes.append((h_out, w_out))
                in_channels = out_channels
            enc_conv_layers.append(nn.Sequential(*enc_block_stack))

            if layer < self.max_depth - 1:
                down_layers.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="downsampler",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[1],
                        stride=2,
                        down_method=downsampler,
                        n_bins=h_in,
                        n_samples=w_in,
                    )
                )
                out_channels *= 2
                h_out, w_out = h_in // 2, w_in // 2
                output_sizes.append((h_out, w_out))

        out_channels = in_channels // 2

        for layer in range(self.max_depth - 1):
            h_in, w_in = output_sizes[-(layer * (block_size + 1)) - 1]
            h_out, w_out = output_sizes[-((layer + 1) * (block_size + 1)) - 1]

            up_layers.append(
                _AEBlock.get_autoencoder_block(
                    block_type="upsampler",
                    dim=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size[2],
                    stride=2,
                    up_method=upsampler,
                    n_bins_in=h_in,
                    n_samples_in=w_in,
                    n_bins_out=h_out,
                    n_samples_out=w_out,
                    scale_factor=2.0 if upsampler != "transpose" else 0,
                    same_padding=same_padding,
                )
            )

            if not use_skip:
                in_channels = out_channels
            dec_block_stack = []
            for _ in range(block_size):
                dec_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="conv",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding="same" if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=self.activation_param,
                        dropout_p=dropout_p,
                    )
                )
                in_channels = out_channels

            dec_conv_layers.append(nn.Sequential(*dec_block_stack))
            in_channels = out_channels
            out_channels //= 2

        self.encoder_conv_layers = enc_conv_layers
        self.decoder_conv_layers = dec_conv_layers
        self.encoder_down = down_layers
        self.decoder_up = up_layers

        self.soft_conv = _SoftConv(
            hidden_channels=hidden_size,
            num_targets=num_targets,
            out_channels=num_channels,
        )

    @abstractmethod
    def encode(
        self, data: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        """Encode method. Projects a sample x onto the latent space."""
        pass

    @abstractmethod
    def decode(
        self, data: torch.FloatTensor, skip_data: List[torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Decode method. Reconstructs a sample x~ from a latent variable z."""
        pass

    @abstractmethod
    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method for all subclasses."""
        pass


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        num_targets: int,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        max_depth: int,
        hidden_size: int,
        kernel_size: Union[Tuple, int],
        same_padding: bool = True,
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        activation: str = "relu",
        leak_constant: Optional[float] = None,
        elu_constant: Optional[float] = None,
        prelu_n_params: Optional[int] = None,
        dropout_p: float = 0,
        use_skip: bool = True,
    ):
        super(AutoEncoder2d, self).__init__()
        if num_targets < 1 or num_targets > 4:
            raise ValueError(
                "Number of targets must be between 1 and 4, but received"
                f" {num_targets}."
            )
        elif num_channels > 2 or num_channels < 1:
            raise ValueError(f"Channels must be 1 (mono) or 2 (stereo).")
        elif hidden_size < 0:
            raise ValueError(f"Hidden size must be at least 1.")
        elif isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 4 - bool(downsampler == "downsample"):
                raise ValueError(
                    "Must specify 4 kernel sizes, but received only"
                    f"{len(kernel_size)}."
                )
        elif block_size < _MIN_BLOCK_SIZE or block_size > _MAX_BLOCK_SIZE:
            raise ValueError(
                f"Block size must be between {_MIN_BLOCK_SIZE} and"
                f" {_MAX_BLOCK_SIZE}, but requested {block_size}"
            )

        self.num_targets = num_targets
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.same_padding = same_padding
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_p = dropout_p
        self.use_skip = use_skip

        # Put a ceiling on the depth of autoencoder.
        self.max_depth = max(
            min(max_depth, int(np.log2(num_bins // hidden_size + 1e-6) + 1)),
            _MIN_AUTOENCODER_DEPTH,
        )
        self.activation_param = leak_constant or elu_constant or prelu_n_params

        enc_conv_layers = nn.ModuleList()
        dec_conv_layers = nn.ModuleList()
        down_layers = nn.ModuleList()
        up_layers = nn.ModuleList()

        output_sizes = [[num_bins, num_samples]]
        in_channels, out_channels = num_channels, hidden_size

        for layer in range(self.max_depth):
            h_in, w_in = output_sizes[-1]
            enc_block_stack = []
            for _ in range(block_size):
                enc_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="conv",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding="same" if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=self.activation_param,
                        dropout_p=dropout_p,
                    )
                )
                if same_padding:
                    h_out, w_out = h_in, w_in
                else:
                    h_out, w_out = _get_conv_output_size(
                        h_in=h_in,
                        w_in=w_in,
                        stride=1,
                        kernel_size=kernel_size[0],
                    )
                output_sizes.append((h_out, w_out))
                in_channels = out_channels
            enc_conv_layers.append(nn.Sequential(*enc_block_stack))

            if layer < self.max_depth - 1:
                down_layers.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="downsampler",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[1],
                        stride=2,
                        down_method=downsampler,
                        n_bins=h_in,
                        n_samples=w_in,
                    )
                )
                out_channels *= 2
                h_out, w_out = h_in // 2, w_in // 2
                output_sizes.append((h_out, w_out))

        out_channels = in_channels // 2

        for layer in range(self.max_depth - 1):
            h_in, w_in = output_sizes[-(layer * (block_size + 1)) - 1]
            h_out, w_out = output_sizes[-((layer + 1) * (block_size + 1)) - 1]

            up_layers.append(
                _AEBlock.get_autoencoder_block(
                    block_type="upsampler",
                    dim=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size[2],
                    stride=2,
                    up_method=upsampler,
                    n_bins_in=h_in,
                    n_samples_in=w_in,
                    n_bins_out=h_out,
                    n_samples_out=w_out,
                    scale_factor=2.0 if upsampler != "transpose" else 0,
                    same_padding=same_padding,
                )
            )

            if not use_skip:
                in_channels = out_channels
            dec_block_stack = []
            for _ in range(block_size):
                dec_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="conv",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding="same" if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=self.activation_param,
                        dropout_p=dropout_p,
                    )
                )
                in_channels = out_channels

            dec_conv_layers.append(nn.Sequential(*dec_block_stack))
            in_channels = out_channels
            out_channels //= 2

        self.encoder_conv_layers = enc_conv_layers
        self.decoder_conv_layers = dec_conv_layers
        self.encoder_down = down_layers
        self.decoder_up = up_layers

        self.soft_conv = _SoftConv(
            hidden_channels=hidden_size,
            num_targets=num_targets,
            out_channels=num_channels,
            num_dims=2
        )

    def encode(
        self, data: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        """Encode."""
        skip_data = []
        for layer in range(len(self.encoder_conv_layers)):
            data = self.encoder_conv_layers[layer](data)
            skip_data.append(data)
            if layer < len(self.encoder_conv_layers) - 1:
                data = self.encoder_down[layer](data)
        return data, skip_data[:-1]

    def decode(
        self, data: torch.FloatTensor, skip_data: List[torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Decode."""
        for layer in range(len(self.decoder_conv_layers)):
            data = self.decoder_up[layer](
                data, output_size=skip_data[-1 - layer].size()
            )
            if self.use_skip:
                data = torch.cat([data, skip_data[-1 - layer]], dim=1)
            data = self.decoder_conv_layers[layer](data)
        return data

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        data, skip_data = self.encode(data)
        data = self.decode(data, skip_data)
        output = self.soft_conv(data)
        return output


class VAE2d(nn.Module):
    def __init__(
        self,
        num_targets: int,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        latent_size: int,
        max_depth: int,
        hidden_size: int,
        kernel_size: Union[Tuple, int],
        same_padding: bool = True,
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        activation: str = "relu",
        leak_constant: Optional[float] = None,
        elu_constant: Optional[float] = None,
        prelu_n_params: Optional[int] = None,
        dropout_p: float = 0,
        use_skip: bool = True,
    ):
        super(VAE2d, self).__init__()
        if num_targets < 0 or num_targets > 4:
            raise ValueError(
                "Number of targets must be between 1 and 4, but received"
                f" {num_targets}"
            )
        elif num_channels > 2 or num_channels < 1:
            raise ValueError(f"Channels must be 1 (mono) or 2 (stereo).")
        elif hidden_size < 0:
            raise ValueError(f"Hidden size must be at least 1.")
        elif isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 4 - bool(downsampler == "downsample"):
                raise ValueError(
                    "Must specify 4 kernel sizes, but received only"
                    f"{len(kernel_size)}."
                )
        elif block_size < _MIN_BLOCK_SIZE or block_size > _MAX_BLOCK_SIZE:
            raise ValueError(
                f"Block size must be between {_MIN_BLOCK_SIZE} and"
                f" {_MAX_BLOCK_SIZE}, but requested {block_size}"
            )

        self.num_targets = num_targets
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.same_padding = same_padding
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_p = dropout_p
        self.use_skip = use_skip

        # Put a ceiling on the depth of autoencoder.
        self.max_depth = max(
            min(max_depth, int(np.log2(num_bins // hidden_size + 1e-6) + 1)),
            _MIN_AUTOENCODER_DEPTH,
        )
        self.activation_param = leak_constant or elu_constant or prelu_n_params

        enc_conv_layers = nn.ModuleList()
        dec_conv_layers = nn.ModuleList()
        down_layers = nn.ModuleList()
        up_layers = nn.ModuleList()

        output_sizes = [[num_bins, num_samples]]
        in_channels, out_channels = num_channels, hidden_size

        for layer in range(self.max_depth):
            h_in, w_in = output_sizes[-1]
            enc_block_stack = []
            for _ in range(block_size):
                enc_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="conv",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding="same" if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=self.activation_param,
                        dropout_p=dropout_p,
                    )
                )
                if same_padding:
                    h_out, w_out = h_in, w_in
                else:
                    h_out, w_out = _get_conv_output_size(
                        h_in=h_in,
                        w_in=w_in,
                        stride=1,
                        kernel_size=kernel_size[0],
                    )
                output_sizes.append((h_out, w_out))
                in_channels = out_channels
            enc_conv_layers.append(nn.Sequential(*enc_block_stack))

            if layer < self.max_depth - 1:
                down_layers.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="downsampler",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[1],
                        stride=2,
                        down_method=downsampler,
                        n_bins=h_in,
                        n_samples=w_in,
                    )
                )
                out_channels *= 2
                h_out, w_out = h_in // 2, w_in // 2
                output_sizes.append((h_out, w_out))

        out_channels = in_channels // 2

        linear_size = output_sizes[-1][0] * output_sizes[-1][1] * in_channels
        self.mu = nn.Linear(linear_size, latent_size)
        self.sigma = nn.Linear(linear_size, latent_size)
        self.eps = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.eps.loc = self.eps.loc.cuda()
            self.eps.scale = self.eps.scale.cuda()

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_size, linear_size), nn.ReLU()
        )

        for layer in range(self.max_depth - 1):
            h_in, w_in = output_sizes[-(layer * (block_size + 1)) - 1]
            h_out, w_out = output_sizes[-((layer + 1) * (block_size + 1)) - 1]

            up_layers.append(
                _AEBlock.get_autoencoder_block(
                    block_type="upsampler",
                    dim=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size[2],
                    stride=2,
                    up_method=upsampler,
                    n_bins_in=h_in,
                    n_samples_in=w_in,
                    n_bins_out=h_out,
                    n_samples_out=w_out,
                    scale_factor=2.0 if upsampler != "transpose" else 0,
                    same_padding=same_padding,
                )
            )

            if not use_skip:
                in_channels = out_channels
            dec_block_stack = []
            for _ in range(block_size):
                dec_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type="conv",
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding="same" if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=self.activation_param,
                        dropout_p=dropout_p,
                    )
                )
                in_channels = out_channels

            dec_conv_layers.append(nn.Sequential(*dec_block_stack))
            in_channels = out_channels
            out_channels //= 2

        self.encoder_conv_layers = enc_conv_layers
        self.decoder_conv_layers = dec_conv_layers
        self.encoder_down = down_layers
        self.decoder_up = up_layers

        self.soft_conv = _SoftConv(
            hidden_channels=hidden_size,
            num_targets=num_targets,
            out_channels=num_channels,
            num_dims=2
        )

    def encode(self, data: torch.FloatTensor) -> Tuple[Tuple, List, Tuple]:
        """Encode."""
        skip_data = []
        for layer in range(len(self.encoder_conv_layers)):
            data = self.encoder_conv_layers[layer](data)
            skip_data.append(data)
            if layer < len(self.encoder_conv_layers) - 1:
                data = self.encoder_down[layer](data)

        batch, n_channels, n_bins, n_samples = data.size()
        input_size = (batch, n_channels, n_bins, n_samples)

        data = data.reshape((batch, n_channels * n_bins * n_samples))
        mu = self.mu(data)
        sigma = torch.exp(self.sigma(data))
        eps = self.eps.sample(sample_shape=sigma.shape)
        latent_data = mu + sigma * eps
        return (latent_data, mu, sigma), skip_data[:-1], input_size

    def decode(
        self,
        data: torch.FloatTensor,
        input_size: Tuple[int],
        skip_data: List[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """Decode."""
        data = self.decoder_linear(data)
        data = data.reshape(input_size)
        for layer in range(len(self.decoder_conv_layers)):
            data = self.decoder_up[layer](
                data, output_size=skip_data[-1 - layer].size()
            )
            if self.use_skip:
                data = torch.cat([data, skip_data[-1 - layer]], dim=1)
            data = self.decoder_conv_layers[layer](data)
        return data

    def forward(
        self, data: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Forward method."""
        latent_output, skip_data, input_size = self.encode(data)
        data = self.decode(
            data=latent_output[0], input_size=input_size, skip_data=skip_data
        )
        output = self.soft_conv(data)
        return output, latent_output[1:]


#
# class LSTMStack(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, depth: int = 3, reshape_input: bool = False):
#         super(LSTMStack, self).__init__()
#         self.hidden_size = hidden_size
#         self.depth = depth
#         self.reshape_input = reshape_input
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, depth=depth, bidirectional=True, batch_first=True)
#
#     def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
#
