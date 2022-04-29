import abc
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from models.layers import _DOWN_LAYERS, _UP_LAYERS, _ACTIVATIONS, _get_activation, _get_conv_padding, \
    _get_transpose_padding, _get_conv_output_size
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Optional, List

_MIN_BLOCK_SIZE = 1
_MAX_BLOCK_SIZE = 10
_MIN_AUTOENCODER_DEPTH = 2
_MAX_AUTOENCODER_DEPTH = 10


class _AEBlock(nn.Module):
    _block_types = {
        'conv',
        'recurrent',
        'upsampler',
        'downsampler',
    }
    """Base class and factory method for all autoencoder blocks.
    
    This class is not meant to be instantiated.
    """

    @staticmethod
    def get_autoencoder_block(
            block_type: str,
            dim: int,
            **kwargs
    ) -> Optional['_AEBlock']:
        """Instantiates an autoencoder block.

        Args:
            block_type (str): Block type to request.
            dim (int): Number of spatial dimensions.

        Returns:
            (optional[_AEBlock]): An autoencoder block.

        Raises:
            ValueError: Raised if the block specifications are invalid.
        """
        if block_type not in _AEBlock._block_types:
            raise ValueError(
                f"{block_type} is not a valid autoencoder block type."
            )
        elif dim != 1 and dim != 2:
            raise ValueError(
                "Spatial dimensions of autoencoder block must be 1d or 2d,"
                f" but received {dim}."
            )

        if block_type == 'conv' and dim == 2:
            return _ConvBlock2d(**kwargs)
        elif block_type == 'conv' and dim == 1:
            return _ConvBlock1d(**kwargs)
        elif block_type == 'recurrent' and dim == 2:
            return _RecurrentBlock2d(**kwargs)
        elif block_type == 'recurrent' and dim == 1:
            return _RecurrentBlock1d(**kwargs)
        elif block_type == 'downsampler' and dim == 2:
            return _DownBlock2d(**kwargs)
        elif block_type == 'downsampler' and dim == 1:
            return _DownBlock1d(**kwargs)
        elif block_type == 'upsampler' and dim == 2:
            return _UpBlock2d(**kwargs)
        elif block_type == 'upsampler' and dim == 1:
            return _UpBlock1d(**kwargs)
        else:
            return None


class _RecurrentBlock2d(_AEBlock):
    def __init__(self, **kwargs):
        super(_RecurrentBlock2d, self).__init__()


class _RecurrentBlock1d(_AEBlock):
    def __init__(self, **kwargs):
        super(_RecurrentBlock1d, self).__init__()


class _ConvBlock2d(_AEBlock):
    """2-dimensional autoencoder convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride length. Default: 1.
        padding (union[tuple, str, int]): Size of the zero-padding. Default: 0.
        activation_fn (str): Activation function. Default: 'relu'.
        batch_norm (bool): Whether to use batch normalization. Default: True.
        use_bias (bool): True if the convolution's bias term should be learned.
            Default: False.
        activation_param (optional[int, float]): Optional activation parameter.
            Default: None.
        dropout_p (float): Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[tuple, str, int] = 'same',
        activation_fn: str = 'relu',
        batch_norm: bool = True,
        use_bias: bool = True,
        activation_param: Optional[Union[int, float]] = None,
        dropout_p: float = 0
    ):
        super(_ConvBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.use_bias = use_bias
        self.activation_param = activation_param
        self.dropout_p = dropout_p

        layers = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.use_bias
            )
        ]

        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.out_channels))
        layers.append(
            _get_activation(self.activation_fn, self.activation_param)
        )
        if self.dropout_p > 0:
            layers.append(nn.Dropout2d(self.dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        return self.layers(data)


class _ConvBlock1d(_AEBlock):
    """1-dimensional autoencoder convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride length. Default: 1.
        padding (union[tuple, str, int]): Size of the zero-padding. Default: 0.
        activation_fn (str): Activation function. Default: 'relu'.
        batch_norm (bool): Whether to use batch normalization. Default: True.
        use_bias (bool): True if the convolution's bias term should be learned.
            Default: False.
        activation_param (optional[int, float]): Optional activation parameter.
            Default: None.
        dropout_p (float): Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[tuple, str, int] = 'same',
        activation_fn: str = 'relu',
        batch_norm: bool = True,
        use_bias: bool = True,
        activation_param: Optional[Union[int, float]] = None,
        dropout_p: float = 0
    ):
        super(_ConvBlock1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.use_bias = use_bias
        self.activation_param = activation_param
        self.dropout_p = dropout_p
        layers = [
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.use_bias
            )
        ]
        if self.batchnorm:
            layers.append(nn.BatchNorm1d(self.out_channels))
        layers.append(
            _get_activation(self.activation_fn, self.activation_param)
        )
        if self.dropout_p > 0:
            layers.append(nn.Dropout(self.dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method.
        """
        return self.layers(data)


class _DownBlock2d(_AEBlock):
    """ 2-dimensional downsampling autoencoder block.

    Args:
        down_method (str): Downsampling method.
        kernel_size (int): Kernel size.
        stride (int): Stride length.
        in_channels (optional[int]): Input channels for 'conv' downsampling.
            Default: None.
        out_channels (optional[int]): Output channels for 'conv' downsampling.
            Default: None.
        n_bins (optional[int]): Number of spatial bins (width) for 'conv'
            downsampling. Default None.
        n_samples (optional[int]: Number of spatial samples (height) for
            'conv' downsampling. Default: None.

    Raises:
        ValueError: Raised when the downsampling method is invalid.
    """

    def __init__(
        self,
        down_method: str,
        kernel_size: int,
        stride: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        n_bins: Optional[int] = None,
        n_samples: Optional[int] = None,
    ):
        super(_DownBlock2d, self).__init__()

        if down_method not in _DOWN_LAYERS:
            raise ValueError(
                f"Downsampling layer must be one of {_DOWN_LAYERS}, but"
                f" received {down_method}."
            )
        elif down_method == 'conv':
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "Input and output channels must be specified for conv"
                    f"downsampling, but received {in_channels} and"
                    f" {out_channels}."
                )
            elif n_bins is None or n_samples is None:
                raise ValueError(
                    f"Spatial dimensions must be specified for conv"
                    f" downsampling, but received {n_bins} and {n_samples}."
                )

        self.down_method = down_method
        self.kernel_size = kernel_size
        self.stride = stride

        if down_method == 'max_pool':
            self.down = nn.MaxPool2d(self.kernel_size, stride=stride)
        elif down_method == 'avg_pool':
            self.down = nn.AvgPool2d(self.kernel_size, stride=stride)
        elif down_method == 'downsample':
            # TODO
            pass
        elif down_method == 'conv':
            padding = _get_conv_padding(
                h_in=n_bins,
                w_in=n_samples,
                h_out=n_bins // 2,
                w_out=n_samples // 2,
                kernel_size=kernel_size
            )
            self.down = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """"""
        return self.down(data)


class _DownBlock1d(_AEBlock):
    """ 1-dimensional downsampling autoencoder block.

    Args:
        down_method (str): Downsampling method.
        kernel_size (int): Kernel size.
        stride (int): Stride length.
        in_channels (Optional[int]): Input channels for 'conv' downsampling.
            Default: None.
        out_channels (Optional[int]): Output channels for 'conv' downsampling.
            Default: None.
        n_samples (Optional[int]: Number of spatial samples (height) for
            'conv' downsampling. Default: None.

    Raises:
        ValueError: Raised when the downsampling method is invalid.
    """

    def __init__(
        self,
        down_method: str,
        kernel_size: int,
        stride: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        n_samples: Optional[int] = None,
    ):
        super(_DownBlock1d, self).__init__()

        if down_method not in _DOWN_LAYERS:
            raise ValueError(
                f"Downsampling layer must be one of {_DOWN_LAYERS},"
                f" but received {down_method}."
            )
        elif down_method == 'conv':
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "Input and output channels must be specified for conv"
                    f"downsampling, but received {in_channels} and"
                    f" {out_channels}."
                )
            elif n_samples is None:
                raise ValueError(
                    f"Spatial dimension must be specified for conv"
                    f" downsampling, but received {n_samples}."
                )

        self.down_method = down_method
        self.kernel_size = kernel_size
        self.stride = stride

        if down_method == 'max_pool':
            self.down = nn.MaxPool1d(self.kernel_size, stride=stride)
        elif down_method == 'avg_pool':
            self.down = nn.AvgPool1d(self.kernel_size, stride=stride)
        elif down_method == 'downsample':
            # TODO
            pass
        elif down_method == 'conv':
            padding, _ = _get_conv_padding(
                h_in=0,
                w_in=n_samples,
                h_out=0,
                w_out=n_samples // 2,
                kernel_size=kernel_size
            )
            self.down = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            self.down = nn.Identity()

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        return self.down(data)


class _UpBlock2d(_AEBlock):
    """ 2-dimensional upsampling autoencoder block.

    Args:
        up_method (str): Downsampling method.
        kernel_size (int): Kernel size.
        stride (int): Stride length.
        n_bins_in (int): Number of input bins.
        n_samples_in (int): Number of input samples.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_bins_out (optional[int]): Number of output bins for transpose conv.
            Default: None.
        n_samples_out (optional[int]): Number of output samples for transpose
            conv. Default: None.
        scale_factor (optional[float]): The scaling factor for interpolation.
            Default: 2.0
        same_padding (bool): Whether to use 'same' padding for the
            post-upsampling conv layer if using interpolation. Default: True.

    Raises:
        ValueError: Raised when the upsampling method is invalid.
    """

    def __init__(
        self,
        up_method: str,
        kernel_size: int,
        stride: int,
        n_bins_in: int,
        n_samples_in: int,
        in_channels: int,
        out_channels: int,
        n_bins_out: Optional[int] = None,
        n_samples_out: Optional[int] = None,
        scale_factor: Optional[float] = 2.0,
        same_padding: bool = True,
    ):
        super(_UpBlock2d, self).__init__()

        if up_method not in _UP_LAYERS:
            raise ValueError(
                f"Upsampling layer must be one of {_UP_LAYERS},"
                f" but received {up_method}."
            )
        elif up_method != 'transpose' and scale_factor is None:
            raise ValueError(
                f"Must specify scale_factor for non-transpose upsampling"
                f" layer, but received {up_method}."
            )
        elif up_method == 'transpose':
            if n_bins_out is None or n_samples_out is None:
                raise ValueError(
                    f"Must specify output shape for transpose upsampling layer,"
                    f"but received {(n_bins_out, n_samples_out)}."
                )
            elif n_bins_out < n_bins_in:
                raise ValueError(
                    f"Output bins {n_bins_out} must be at least"
                    f" {n_bins_in}."
                )
            elif n_samples_out < n_samples_in:
                raise ValueError(
                    f"Output samples {n_samples_out} must be at least"
                    f" {n_samples_in}."
                )

        self.down_method = up_method
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_bins_in = n_bins_in
        self.n_samples_in = n_samples_in
        self.in_channels = in_channels
        self.out_channels = out_channels

        if up_method == 'transpose':
            padding = _get_transpose_padding(
                h_in=n_bins_in,
                w_in=n_samples_in,
                h_out=n_bins_out,
                w_out=n_samples_out,
                stride=stride,
                kernel_size=kernel_size
            )
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.is_transpose = True
        else:
            scale_factor = scale_factor
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=up_method),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='same' if same_padding else 0
                )
            )
            self.is_transpose = False

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        return self.down(data)


class _UpBlock1d(_AEBlock):
    """ 1-dimensional upsampling autoencoder block.

    Args:
        up_method (str): Downsampling method.
        kernel_size (int): Kernel size.
        stride (int): Stride length.
        n_samples_in (int): Number of input samples.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_samples_out (optional[int]): Number of output samples for transpose
            conv. Default: None.
        scale_factor (optional[float]): The scaling factor for interpolation.
            Default: 2.0
        same_padding (bool): Whether to use 'same' padding for the
            post-upsampling conv layer if using interpolation. Default: True.

    Raises:
        ValueError: Raised when the upsampling method is invalid.
    """

    def __init__(
        self,
        up_method: str,
        kernel_size: int,
        stride: int,
        n_samples_in: int,
        in_channels: int,
        out_channels: int,
        n_samples_out: Optional[int] = None,
        scale_factor: Optional[float] = 2.0,
        same_padding: bool = True,
    ):
        super(_UpBlock1d, self).__init__()

        if up_method not in _UP_LAYERS:
            raise ValueError(
                f"Upsampling layer must be one of {_UP_LAYERS},"
                f" but received {up_method}."
            )
        elif up_method != 'transpose' and scale_factor is None:
            raise ValueError(
                f"Must specify scale_factor for non-transpose upsampling"
                f" layer, but received {up_method}."
            )
        elif up_method == 'transpose':
            if n_samples_out is None:
                raise ValueError(
                    f"Must specify output shape for transpose upsampling layer,"
                    f"but received {n_samples_out}."
                )
            elif n_samples_out < n_samples_in:
                raise ValueError(
                    f"Output samples {n_samples_out} must be at least"
                    f" {n_samples_in}."
                )

        self.down_method = up_method
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_samples_in = n_samples_in
        self.in_channels = in_channels
        self.out_channels = out_channels

        if up_method == 'transpose':
            _, padding = _get_transpose_padding(
                h_in=0,
                w_in=n_samples_out,
                h_out=0,
                w_out=n_samples_out,
                stride=stride,
                kernel_size=kernel_size
            )
            self.up = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.is_transpose = True
        else:
            scale_factor = scale_factor
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=up_method),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='same' if same_padding else 0
                )
            )
            self.is_transpose = False

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        return self.down(data)


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        max_depth: int,
        hidden_size: int,
        kernel_size: Union[Tuple, int],
        same_padding: bool = True,
        block_size: int = 3,
        downsampler: str = 'max_pool',
        upsampler: str = 'transpose',
        batch_norm: bool = True,
        activation: str = 'relu',
        leak_constant: Optional[float] = None,
        elu_constant: Optional[float] = None,
        prelu_n_params: Optional[int] = None,
        dropout_p: float = 0,
        use_skip: bool = True
    ):
        super(AutoEncoder2d, self).__init__()
        if num_channels > 2 or num_channels < 1:
            raise ValueError(f"Channels must be 1 (mono) or 2 (stereo).")
        # Put a ceiling on the depth of autoencoder.
        max_depth = max(min(
            max_depth, int(np.log2(num_bins // hidden_size + 1e-6) + 1)),
            _MIN_AUTOENCODER_DEPTH
        )
        if hidden_size < 0:
            raise ValueError(f"Hidden size must be at least 1.")
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        else:
            if len(kernel_size) != 4 - bool(downsampler == 'downsample'):
                raise ValueError(
                    "Must specify 4 kernel sizes, but received only"
                    f"{len(kernel_size)}."
                )
        if block_size < _MIN_BLOCK_SIZE or block_size > _MAX_BLOCK_SIZE:
            raise ValueError(
                f"Block size must be between {_MIN_BLOCK_SIZE} and"
                f" {_MAX_BLOCK_SIZE}, but requested {block_size}"
            )
        activation_param = leak_constant or elu_constant or prelu_n_params

        self.num_bins = num_bins
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.max_depth = max_depth
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

        enc_conv_layers = nn.ModuleList()
        dec_conv_layer = nn.ModuleList()
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
                        block_type='conv',
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding='same' if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=activation_param,
                        dropout_p=dropout_p
                    )
                )
                if same_padding:
                    h_out, w_out = h_in, w_in
                else:
                    h_out, w_out = _get_conv_output_size(
                        h_in=h_in,
                        w_in=w_in,
                        stride=1,
                        kernel_size=kernel_size[0]
                    )
                output_sizes.append((h_out, w_out))
                in_channels = out_channels
            enc_conv_layers.append(nn.Sequential(*enc_block_stack))

            if layer < self.max_depth - 1:
                down_layers.append(
                    _AEBlock.get_autoencoder_block(
                        block_type='downsampler',
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[1],
                        stride=2,
                        down_method=downsampler,
                        n_bins=h_in,
                        n_samples=w_in
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
                    block_type='upsampler',
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
                    scale_factor=2.0 if upsampler != 'transpose' else 0,
                    same_padding=same_padding
                )
            )

            if not use_skip:
                in_channels = out_channels
            dec_block_stack = []
            for _ in range(block_size):
                dec_block_stack.append(
                    _AEBlock.get_autoencoder_block(
                        block_type='conv',
                        dim=2,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size[0],
                        stride=1,
                        padding='same' if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        use_bias=not batch_norm,
                        activation_param=activation_param,
                        dropout_p=dropout_p
                    )
                )
                in_channels = out_channels

            dec_conv_layer.append(nn.Sequential(*dec_block_stack))
            in_channels = out_channels
            out_channels //= 2





class BlockStack(nn.Module):
    """Base class for autoencoder blocks.

    Args:
        block_stack (List[nn.Module]): The stack of convolutional blocks.
        spatial_layer (nn.Module): And upsampling or downsampling layer.
    """

    def __init__(
            self,
            block_stack: List[nn.Module],
            spatial_layer: nn.Module
    ):
        super(BlockStack, self).__init__()
        self._block_stack = nn.Sequential(*block_stack)
        self._spatial_layer = spatial_layer


class EncoderStack(BlockStack):
    """"""

    def __init__(
            self,
            encoder_stack: List[nn.Module],
            downsampler: nn.Module
    ):
        super(EncoderStack, self).__init__(encoder_stack, downsampler)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """"""
        data = self.block_stack(data)
        output = self.spatial_layer(data)
        return output


class EncoderSkipStack(EncoderStack):
    """"""

    def __init__(self, **kwargs):
        super(EncoderSkipStack, self).__init__(**kwargs)

    def forward(
            self,
            data: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """"""
        skip_data = self.block_stack(data)
        output = self.spatial_layer(skip_data)
        return output, skip_data


class DecoderStack(BlockStack):
    """"""

    def __init__(
            self,
            decoder_stack: List[nn.Module],
            upsampler: nn.Module
    ):
        super(DecoderStack, self).__init__(decoder_stack, upsampler)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """"""
        data = self.spatial_layer(data)
        output = self.block_stack(data)
        return output


class DecoderSkipStack(BlockStack):
    """"""

    def __init__(self, **kwargs):
        super(DecoderSkipStack, self).__init__(**kwargs)

    def forward(
            self,
            data: torch.FloatTensor,
            skip_data: torch.FloatTensor
    ) -> torch.FloatTensor:
        """"""
        data = self.spatial_layer(data)
        output = self.block_stack(data)
        return output

#
# class ConvStack1d(AutoEncoderBlock):
#     pass
