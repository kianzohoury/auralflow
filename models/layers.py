
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as functional
from pprint import pprint
from collections import OrderedDict
from typing import Optional, Union, List, Tuple
from abc import ABCMeta, abstractmethod

_MIN_BLOCK_SIZE = 1
_MAX_BLOCK_SIZE = 10
_MIN_AUTOENCODER_DEPTH = 2
_MAX_AUTOENCODER_DEPTH = 10
_ACTIVATIONS = {
    'relu',
    'leaky_relu',
    'sigmoid',
    'tanh',
    'elu',
    'prelu',
    'glu'
}

_DOWN_LAYERS = {
    'max_pool',
    'avg_pool',
    'conv',
    'downsample',
}
_UP_LAYERS = {
    'transpose',
    'upsample',
}
_INTERPOLATE_MODES = {
    'nearest',
    'linear',
    'bilinear',
    'trilinear',
    'cubic'
}


def get_conv_output_size(
    h_in: int,
    w_in: int,
    stride: int,
    kernel_size: int
) -> Tuple[int, int]:
    """Computes the non-zero padded output of a conv layer.

    Returns:
        (tuple): Output size.
    """
    h_out = math.floor((h_in - kernel_size // stride) + 1)
    w_out = math.floor((w_in - kernel_size // stride) + 1)
    assert h_out >= 0 and w_out >= 0
    return h_out, w_out


def get_transpose_padding(
    h_in: int,
    w_in: int,
    h_out: int,
    w_out: int,
    stride: int,
    kernel_size: int
) -> Tuple[int, int]:
    """Computes the required transpose conv padding for a target shape.

    Returns:
        (tuple): Transpose padding.
    """
    h_pad = math.ceil((kernel_size - h_out + stride * (h_in - 1)) / 2)
    w_pad = math.ceil((kernel_size - w_out + stride * (w_in - 1)) / 2)
    assert h_pad >= 0 and w_pad >= 0
    return h_pad, w_pad


def get_conv_padding(
    h_in: int,
    w_in: int,
    h_out: int,
    w_out: int,
    kernel_size: int
) -> Tuple[int, int]:
    """Computes the required conv padding.

    Returns:
        (tuple): Convolutional padding.
    """
    h_pad = max(0, math.ceil((2 * h_out - 2 + kernel_size - h_in) / 2))
    w_pad = max(0, math.ceil((2 * w_out - 2 + kernel_size - w_in) / 2))
    assert h_pad >= 0 and w_pad >= 0
    return h_pad, w_pad


def get_activation(
    activation_fn: str,
    param: Optional[int, float] = None
) -> nn.Module:
    """Returns an instantiation of an activation layer.

    Args:
        activation_fn (str): Name of the activation function.
        param (int or float or None): An optional parameter.

    Returns:
        (nn.Module): The activation function.

    Raises:
        ValueError: Raised when the activation is not valid or not accepted.
    """
    if activation_fn in _ACTIVATIONS:
        if activation_fn == 'relu':
            return nn.ReLU()
        elif activation_fn == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=param)
        elif activation_fn == 'elu':
            return nn.ELU(alpha=param)
        elif activation_fn == 'prelu':
            return nn.PReLU(num_parameters=param)
        elif activation_fn == 'glu':
            return nn.GLU()
        elif activation_fn == 'sigmoid':
            return nn.Sigmoid()
        elif activation_fn == 'tanh':
            return nn.Tanh()
    raise ValueError(
        f"Activation function must be one of {_ACTIVATIONS},"
        f" but received {activation_fn}."
    )


class DownSample(nn.Module):
    """Wrapper class for Pytorch's interpolation module.

    Args:
        scale_factor (float): Factor to scale down by. Default: 0.5.

    Raises:
        ValueError: Raised when the scaling factor is impossible.
    """
    def __init__(self, scale_factor: float = 0.5):
        super(DownSample, self).__init__()
        if scale_factor < 0 or scale_factor > 1.0:
            raise ValueError(
                "Scale factor must be between 0 and 1,"
                f"but received a value of {scale_factor}."
            )
        self._scale_factor = scale_factor

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        output = functional.interpolate(
            input=data,
            size=self._output_size,
        )
        return output


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        num_bins: int,
        num_samples: int,
        num_channels: int,
        max_depth: int,
        hidden_size: int = 16,
        kernel_sizes: Union[Tuple, int] = 3,
        same_padding: bool = True,
        block_size: int = 1,
        downsampler: str = 'max_pool',
        upsampler: str = 'transpose',
        batch_norm: bool = True,
        activation: str = 'relu',
        leak: Optional[float] = None,
        dropout_p: float = 0,
        num_dropouts: int = 0,
        use_skip: bool = True
    ):
        super(AutoEncoder2d, self).__init__()
        self.num_bins = num_bins
        self.num_samples = num_samples
        assert 1 <= num_channels <= 2, (
            f"Channels must be 1 (mono) or 2 (stereo)."
        )
        self.num_channels = num_channels
        # Correct the depth if needed.
        self.max_depth = max(min(
            max_depth, int(np.log2(num_bins // hidden_size + 1e-6) + 1)),
            _MIN_AUTOENCODER_DEPTH
        )
        assert hidden_size > 0, f"Hidden size must be at least 1."
        self.hidden_size = hidden_size
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 4
        else:
            assert len(kernel_sizes) == 4, (
                "Must specify 4 kernel sizes, but received only"
                f"{len(kernel_sizes)}."
            )

        assert 0 < block_size < 10, f"Block size"
        assert _MIN_BLOCK_SIZE <= block_size <= _MAX_BLOCK_SIZE, (
            f"Block size must be between {_MIN_BLOCK_SIZE} and"
            f" {_MAX_BLOCK_SIZE}, but requested {block_size}"
        )
        assert upsampler in _UP_LAYERS, (
            f"Upsampler must be one of {_UP_LAYERS}, but received"
            f" {upsampler}."
        )

        encoder = nn.ModuleList()
        decoder = nn.ModuleList()
        output_sizes = [[num_bins, num_samples]]
        in_channels, out_channels = num_channels, hidden_size

        for layer in range(self.max_depth):
            encoder_block = []

            for sub_layer in range(block_size):
                is_last = sub_layer == self.max_depth - 1
                encoder_block.append(
                    EncoderBlock2d(
                        in_channels=out_channels if sub_layer else in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[0],
                        padding='same' if same_padding else 0,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        bias=not batch_norm,
                        leak=leak,
                        down_method=downsampler if is_last else None,
                        down_kernel_size=kernel_sizes[1] if is_last else None,
                        num_bins=output_sizes[-1][0],
                        num_samples=output_sizes[-1][1]
                    )
                )

                output_sizes.append(
                    encoder_block[-1].output_size((num_bins, num_samples))
                )
            encoder.append(nn.Sequential(*encoder_block))
            in_channels *= 2
            out_channels *= 2

        for layer in range(self.max_depth):
            decoder_block = []

            for sub_layer in range(block_size):
                is_first = sub_layer == 0
                if is_first:
                    h_in, w_in = output_sizes[-(layer + 1) * block_size]
                    h_out, w_out = output_sizes[-(layer + 1) * block_size - 1]
                    padding = get_transpose_padding(
                        h_in=h_in,
                        w_in=w_in,
                        h_out=h_out,
                        w_out=w_out,
                        stride=2,
                        kernel_size=kernel_sizes[3]
                    )
                else:
                    padding = 'same'

                decoder_block.append(
                    DecoderBlock2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[2],
                        padding=padding,
                        activation_fn=activation,
                        batch_norm=batch_norm,
                        bias=not batch_norm,
                        up_method=upsampler if is_first else None,
                        up_kernel_size=kernel_sizes[3] if is_first else None,
                        upsample_mode=upsampler,
                        use_skip=use_skip,
                        dropout_p=dropout_p
                    )
                )

            decoder.append(nn.Sequential(*decoder_block))
            in_channels //= 2
            out_channels //= 2

        self.encoder = encoder
        self.decoder = decoder
        # self.encoder.fo

    # def forward(self, ):

#
# class EncoderModule(nn.Module):
#     def __init__(self, encoder_list: nn.ModuleList):
#         self.encoder =

class AutoEncoderBlock(metaclass=ABCMeta, nn.Module):
    """Base class for autoencoder layers.

    EncoderBlock1d, EncoderBlock2d, DecoderBlock1d and DecoderBlock2d
    implement this class.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size or shape of the kernel. Default: 5.
        stride (int): Stride length. Default: 1.
        padding (tuple or str or int or None): Size of zero-padding.
            Default: None.
        activation_fn (str): Activation function. Default: 'relu'.
        batch_norm (bool): Whether to apply batch normalization. Default: True.
        bias (bool): True if the convolution's bias term should be learned.
            Default: False.
        leak (float or None): Negative slope value if using leaky ReLU.
            Default: 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: Optional[tuple, str, int] = 0,
        activation_fn: str = 'relu',
        batch_norm: bool = True,
        bias: bool = False,
        leak: Optional[float] = None,
    ):
        super(AutoEncoderBlock, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation_fn = activation_fn
        self._batch_norm = batch_norm
        self._bias = bias
        self._leak = leak


class EncoderBlock2d(AutoEncoderBlock):
    """Encoder block.

    By default, this block doubles the feature dimension (channels)
    but halves the spatial dimensions of the input (freq and time).

    Args:
        down_method (str or None): The downsampling method.
        down_kernel_size

    Raises:
        ValueError: Raised if the downsampling method or arguments are invalid.

    Examples:
        >>> from models.layers import EncoderBlock2d
        >>> encoder = EncoderBlock2d(
        ...     in_channels=16,
        ...     out_channels=32,
        ...     kernel_size=5,
        ...     stride=2,
        ...     padding=2,
        ...     activation_fn='relu',
        ...     batch_norm=True,
        ...     down_method='max_pool'
        ... )
        >>> data = torch.rand((1, 16, 4, 1))
        >>> output = encoder(data)
    """
    def __init__(
        self,
        down_method: Optional[str],
        down_kernel_size: Optional[int] = None,
        num_bins: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        super(EncoderBlock2d, self).__init__(**kwargs)
        self.conv = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            bias=self._bias
        )
        if self._batch_norm:
            self.batchnorm = nn.BatchNorm2d(self._out_channels)
        else:
            self.batchnorm = nn.Identity()
        self.activation = get_activation(self._activation_fn, self._leak)
        if down_method is None:
            self.down = nn.Identity()
        elif down_method not in _DOWN_LAYERS:
            raise ValueError(
                f"Downsampler must be one of {_DOWN_LAYERS}, but received"
                f" {down_method}."
            )
        elif down_method != 'downsample' and down_kernel_size is None:
            raise ValueError(
                f"Kernel size for {down_method} must be specified."
            )
        elif down_method == 'max_pool':
            self.down = nn.MaxPool2d(self._down_kernel_size)
        elif down_method == 'avg_pool':
            self.down = nn.AvgPool2d(self._down_kernel_size)
        else:
            if num_bins is None or num_samples is None:
                raise ValueError(
                    "Spatial dimensions must be specified for convolutional"
                    " downsampling."
                )
            down_padding = get_conv_padding(
                h_in=num_bins,
                w_in=num_samples,
                h_out=num_bins // 2,
                w_out=num_bins // 2,
                kernel_size=down_kernel_size
            )
            self.down = nn.Conv2d(
                in_channels=self._out_channels,
                out_channels=self._out_channels,
                kernel_size=down_kernel_size,
                stride=2,
                padding=down_padding,
            )

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
        Returns:
            (tensor): Downsampled feature map.
        """
        data = self.conv(data)
        data = self.batchnorm(data)
        data = self.activation(data)
        output = self.down(data)
        return output

    def output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Returns the shape of the encoder's output."""
        test_data = torch.rand((1, self._in_channels, *input_size)).float()
        output = self.forward(test_data)
        h_out, w_out = output[-2:]
        return h_out, w_out


class DecoderBlock2d(AutoEncoderBlock):
    """Encoder block.

    By default, this block doubles the feature dimension (channels)
    but halves the spatial dimensions of the input (freq and time).

    Args:
        down_method (str or None): The downsampling method.
        down_kernel_size

    Raises:
        ValueError: Raised if the downsampling method or arguments are invalid.

    Examples:
        >>> from models.layers import DecoderBlock2d
        >>> decoder = DecoderBlock2d(
        ...     in_channels=16,
        ...     out_channels=1,
        ...     kernel_size=5,
        ...     stride=2,
        ...     padding=2,
        ...     activation_fn='relu',
        ...     up_method='transpose',
        ...     up_kernel_size=5,
        ...     dropout_p=0.4
        ... )
        >>> data = torch.rand((1, 16, 4, 1))
        >>> out = decoder(data, output_size=(1, 1, 8, 2))
    """
    def __init__(
        self,
        up_method: Optional[str],
        up_kernel_size: Optional[int] = None,
        upsample_mode: Optional[str] = None,
        use_skip: bool = True,
        dropout_p: float = 0,
        **kwargs
    ):
        super(DecoderBlock2d, self).__init__(**kwargs)
        self._up_method = up_method
        if up_method is None:
            self.up = nn.Identity()
        elif up_kernel_size is None:
            raise ValueError(
                f"Kernel size for {up_method} must be specified."
            )
        elif up_method == 'upsample':
            if upsample_mode is None:
                raise ValueError(
                    "Upsampling requires a mode to be specified from"
                    f" {_INTERPOLATE_MODES}."
                )
            self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._padding
            )
        if self._batch_norm:
            self.batchnorm = nn.BatchNorm2d(self._out_channels)
        else:
            self.batchnorm = nn.Identity()
        self.activation = get_activation(self._activation_fn)
        self._use_skip = use_skip
        if dropout_p > 0:
            self.dropout = nn.Dropout2d(dropout_p)
        else:
            self.dropout = nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=self._out_channels * (1 + bool(use_skip)),
            out_channels=self._out_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            bias=self._bias
        )

    def forward(
        self,
        data: torch.Tensor,
        skip_data: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
        Returns:
            (tensor): Upsampled feature map.
        """
        if self._up_method == 'transpose':
            data = self.up(data, output_size=skip_data.size())
        else:
            data = self.up(data)
        if self._use_skip:
            data = torch.cat([data, skip_data], dim=1)
        data = self.conv(data)
        data = self.batchnorm(data)
        data = self.activation(data)
        output = self.dropout(data)
        return output







class DecoderBlock(nn.Module):
    """Fully-customizable decoder block layer for base separation models.

    By default, this block doubles the spatial dimensions (freq and time) but
    halves the feature dimension (channels). Corresponds with the encoder at
    the same depth level in the encoder/decoder architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int or None): Number of output channels.
            If None, defaults to 2 * in_channels.
        kernel_size (int or tuple): Size or shape of the convolutional filter.
            Default: 5.
        stride (int or tuple): Size or shape of the stride. Default: 2.
        padding (int or tuple): Size of zero-padding added to each side.
            Default: 2.
        dropout_p (float): Dropout probability. Default: 0.
        skip_block (bool): True if block has a skip connection, false otherwise.
            Default: True.
        activation_fn (str or None): Activation function. Default: 'relu'.
        batch_norm (bool): Whether to apply batch normalization. Default: True.
        bias (bool): Whether to include a bias term in the conv layer.
            Default: False.

    Example:
        >>> from models.layers import DecoderBlock
        >>> decoder = DecoderBlock(16, 1, 5, 2, 2, activation_fn='relu')
        >>> data = torch.rand((1, 16, 4, 1))
        >>> out = decoder(data, output_size=(1, 1, 8, 2))
"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Union[int, tuple] = 5,
        stride: Union[int, tuple] = 2,
        padding: Union[int, str] = 2,
        dropout_p: float = 0,
        skip_block: bool = True,
        activation_fn: Optional[str] = 'relu',
        batch_norm: bool = True,
        bias: bool = False,
    ):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels = in_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.skip_block = skip_block
        self.bias = bias
        self.dropout_p = dropout_p
        self.convT = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        if batch_norm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        else:
            self.batchnorm = nn.Identity()
        if dropout_p > 0:
            self.dropout = nn.Dropout2d(dropout_p)
        else:
            self.dropout = nn.Identity()

    def forward(
        self,
        data: torch.Tensor,
        output_size: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
            output_size (torch.Size): Output size of this layer's output.
        Returns:
            (tensor): Output feature map.
        """
        if output_size:
            x = self.convT(data, output_size=output_size)
        else:
            x = self.convT(data)
        x = self.batchnorm(x)
        x = self.activation(x)
        output = self.dropout(x)
        return output




encoder_scheme = [
    {
        'in_channels': 16,
        'out_channels': 32,
        'kernel_size': 5,
        'stride': 2,
        'padding': 2,
        'down': 'conv',
        'batchnorm': True,
        'activation': 'relu',
        'leak': 0.2,
        'bias': False
     }
]

decoder_scheme = [
    {
        'in_channels': 16,
        'out_channels': 32,
        'kernel_size': 8,
        'stride': 4,
        'padding': 4,
        'up': 'transpose',
        'batchnorm': True,
        'activation': 'relu',
        'dropout_p'
        'bias': False,
        'skip_block': True
    }
]



class StackedBlock(nn.Module):
    """Stacks multiple encoder/decoder blocks together.

    Args:
        scheme (List[dict]): Multi-layered block scheme.

    Examples
        >>> from models.layers import StackedBlock
        >>> scheme = [{'in_channels': 16, 'out_channels': 32,
        ... 'batch_norm': True, 'activation_fn': 'relu'}]
        >>> encoder = StackedBlock(scheme)
        >>> data = torch.rand((1, 16, 4, 1))
        >>> output = encoder(data)

    """
    _keys = {
        'in_channels',
        'out_channels',
        'kernel_size',
        'stride',
        'padding',
        'batch_norm',
        'activation_fn',
        'leak',
        'dropout_p',
        'max_pool',
        'bias',
    }
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 h_in: int,
                 w_in: int,
                 h_out: int,
                 w_out: int,
                 scheme: LayerNode,
                 skip_connections: bool = True,
                 last: bool = False,
                 is_encoder: bool = True,
                 first_decoder: bool = False,
                 use_dropout: bool = True,
                 skip_last: bool = True
                 ):
        super(StackedBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        print(scheme)
        down, up = [], []
        conv_stack = []
        if is_encoder:
            layers = scheme['conv_stack'] + scheme['downsampling_stack']
            split_index = len(scheme['conv_stack'])
        else:
            layers = scheme['upsampling_stack'] + scheme['conv_stack']
            split_index = len(scheme['upsampling_stack'])
        for i, layer in enumerate(layers):
            layer_type = layer['layer_type']
            if layer_type == 'conv':
                kernel_size = layer['param']
                if is_encoder:
                    if layer.get('down', False):
                        padding = get_conv_padding(
                            h_in=h_in,
                            w_in=w_in,
                            h_out=h_out,
                            w_out=w_out,
                            kernel_size=kernel_size
                        )
                        stride = 2
                    else:
                        padding = 'same'
                        stride = 1
                else:
                    if skip_connections and len(conv_stack) == 0 and not skip_last:
                        in_channels = in_channels * 2

                    padding = 'same'
                    stride = 1

                layer_module = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )

                in_channels = out_channels

            elif layer_type == 'transpose_conv':
                kernel_size = layer['param']
                padding = get_transpose_padding(
                    h_in=h_in,
                    w_in=w_in,
                    h_out=h_out,
                    w_out=w_out,
                    stride=2,
                    kernel_size=kernel_size
                )
                print(h_in, w_in, h_out, w_out, padding)
                print(in_channels)
                if skip_connections and not first_decoder and len(scheme['conv_stack']) == 0:
                    in_channels *= 2

                layer_module = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding
                )

                in_channels = out_channels
            elif layer_type == 'upsample':
                layer_module = nn.Upsample(
                    size=torch.Size((h_in, w_in)),
                    scale_factor=2
                )
            elif layer_type == 'max_pool':
                kernel_size = layer['param']
                layer_module = nn.MaxPool2d(kernel_size)
            elif layer_type == 'batch_norm':
                layer_module = nn.BatchNorm2d(out_channels)
            elif layer_type == 'dropout' and use_dropout:
                dropout_p = layer['param']
                layer_module = nn.Dropout2d(dropout_p)
            elif layer_type in {'relu', 'leaky_relu', 'sigmoid', 'tanh'}:
                layer_module = get_activation(layer_type, layer['param'])
            else:
                layer_module = nn.Identity()


            if is_encoder:
                print(i, 123, layer_type, split_index)
                if i >= split_index:
                    down.append(layer_module)
                else:
                    conv_stack.append(layer_module)
            else:
                if i < split_index:
                    up.append(layer_module)
                else:
                    conv_stack.append(layer_module)



        self.conv_stack = nn.Sequential(*conv_stack)

        if is_encoder:
            self.down = DownSampler(down)
        else:
            self.deconv_pre = up[0]
            self.deconv_post = nn.Sequential(*up[1:])

# class StackedBlock(nn.Module):
#     """Stacks multiple encoder/decoder blocks together.
#
#     Args:
#         scheme (List[dict]): Multi-layered block scheme.
#
#     Examples
#         >>> from models.layers import StackedBlock
#         >>> scheme = [{'in_channels': 16, 'out_channels': 32,
#         ... 'batch_norm': True, 'activation_fn': 'relu'}]
#         >>> encoder = StackedBlock(scheme)
#         >>> data = torch.rand((1, 16, 4, 1))
#         >>> output = encoder(data)
#
#     """
#     _keys = {
#         'in_channels',
#         'out_channels',
#         'kernel_size',
#         'stride',
#         'padding',
#         'batch_norm',
#         'activation_fn',
#         'leak',
#         'dropout_p',
#         'max_pool',
#         'bias',
#     }
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  h_in: int,
#                  w_in: int,
#                  h_out: int,
#                  w_out: int,
#                  scheme: dict,
#                  skip_connections: bool = True,
#                  last: bool = False, is_encoder: bool = True,
#                  first_decoder: bool = False,
#                  use_dropout: bool = True,
#                  skip_last: bool = True
#                  ):
#         super(StackedBlock, self).__init__()
#         assert len(scheme) > 0, \
#             f"Must specify a non-empty block scheme, but received {scheme}."
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         down, up = [], []
#         conv_stack = []
#         if is_encoder:
#             layers = scheme['conv_stack'] + scheme['downsampling_stack']
#             split_index = len(scheme['conv_stack'])
#         else:
#             layers = scheme['upsampling_stack'] + scheme['conv_stack']
#             split_index = len(scheme['upsampling_stack'])
#         for i, layer in enumerate(layers):
#             layer_type = layer['layer_type']
#             if layer_type == 'conv':
#                 kernel_size = layer['param']
#                 if is_encoder:
#                     if layer.get('down', False):
#                         padding = get_conv_padding(
#                             h_in=h_in,
#                             w_in=w_in,
#                             h_out=h_out,
#                             w_out=w_out,
#                             kernel_size=kernel_size
#                         )
#                         stride = 2
#                     else:
#                         padding = 'same'
#                         stride = 1
#                 else:
#                     if skip_connections and len(conv_stack) == 0 and not skip_last:
#                         in_channels = in_channels * 2
#
#                     padding = 'same'
#                     stride = 1
#
#                 layer_module = nn.Conv2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     stride=stride,
#                     padding=padding
#                 )
#
#                 in_channels = out_channels
#
#             elif layer_type == 'transpose_conv':
#                 kernel_size = layer['param']
#                 padding = get_transpose_padding(
#                     h_in=h_in,
#                     w_in=w_in,
#                     h_out=h_out,
#                     w_out=w_out,
#                     stride=2,
#                     kernel_size=kernel_size
#                 )
#                 print(h_in, w_in, h_out, w_out, padding)
#                 print(in_channels)
#                 if skip_connections and not first_decoder and len(scheme['conv_stack']) == 0:
#                     in_channels *= 2
#
#                 layer_module = nn.ConvTranspose2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding
#                 )
#
#                 in_channels = out_channels
#             elif layer_type == 'upsample':
#                 layer_module = nn.Upsample(
#                     size=torch.Size((h_in, w_in)),
#                     scale_factor=2
#                 )
#             elif layer_type == 'max_pool':
#                 kernel_size = layer['param']
#                 layer_module = nn.MaxPool2d(kernel_size)
#             elif layer_type == 'batch_norm':
#                 layer_module = nn.BatchNorm2d(out_channels)
#             elif layer_type == 'dropout' and use_dropout:
#                 dropout_p = layer['param']
#                 layer_module = nn.Dropout2d(dropout_p)
#             elif layer_type in {'relu', 'leaky_relu', 'sigmoid', 'tanh'}:
#                 layer_module = get_activation(layer_type, layer['param'])
#             else:
#                 layer_module = nn.Identity()
#
#
#             if is_encoder:
#                 print(i, 123, layer_type, split_index)
#                 if i >= split_index:
#                     down.append(layer_module)
#                 else:
#                     conv_stack.append(layer_module)
#             else:
#                 if i < split_index:
#                     up.append(layer_module)
#                 else:
#                     conv_stack.append(layer_module)
#
#
#
#         self.conv_stack = nn.Sequential(*conv_stack)
#
#         if is_encoder:
#             self.down = DownSampler(down)
#         else:
#             self.deconv_pre = up[0]
#             self.deconv_post = nn.Sequential(*up[1:])

class DownSampler(nn.Module):
    def __init__(self, layers):
        super(DownSampler, self).__init__()
        self.down = nn.Sequential(*layers)

    def forward(self, data):
        return self.down(data)


class StackedEncoderBlock(StackedBlock):
    def __init__(self, **kwargs):
        super(StackedEncoderBlock, self).__init__(**kwargs)

    def forward(self, data):
        encoding = self.conv_stack(data)
        output = self.down(encoding)
        return output, encoding


class StackedDecoderBlock(StackedBlock):
    def __init__(self, **kwargs):
        super(StackedDecoderBlock, self).__init__(**kwargs)

# class StackedDecoderBlock(StackedBlock):
#     def __init__(self, **kwargs):
#         super(StackedDecoderBlock, self).__init__(**kwargs)
#
#     def forward(self, data: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
#         """Forward method.
#
#         Args:
#             data (tensor): Input feature map.
#             output_size (torch.Size): Output size of this layer's output.
#         Returns:
#             (tensor): Output feature map.
#         """
#         x = self.up(data, output_size=output_size)
#         output = self.conv_stack(x)
#         return output

# Very slow.
class GLU2d(nn.Module):
    """GLU adapted for 2-dimensional spatial reduction.

    Args:
        input_size (int): Number of features in the last dimension.
    """
    def __init__(self, input_size):
        super(GLU2d, self).__init__()
        self.linear = nn.Linear(input_size, input_size // 2)
        self.glu = nn.GLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        assert len(data.shape) == 4, \
            (f"Shape of input must be 3-dimensional, "
             f"but received input of size {len(data.shape) - 1}.")
        n, c, w, h = data.shape
        data = data.flatten(start_dim=-2)
        data = self.linear(data)
        out = self.glu(data)
        out = out.reshape((n, c, w // 2, h // 2))
        return out

#
# class StackedDecoderBlock(nn.Module):
#     """Stacks multiple decoder blocks together.
#
#     Useful for constructing larger and more complex decoder blockers.
#
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int or None): Number of output channels. If None,
#             defaults to 2 * in_channels.
#         layers (int): Number of decoder blocks to stack.
#         kernel_size (int or tuple): Size or shape of the convolutional filter.
#             Default: 5.
#         upsampling_method (str): Type of upsampling. Default: 'transposed'.
#         dropout_p (float): Dropout probability. Default: 0.
#         skip_block (bool): True if block has a skip connection, false otherwise.
#             Default: True.
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: Optional[int] = None,
#         layers=1,
#         scheme: Optional[dict] = None,
#         kernel_size: Union[int, tuple] = 5,
#         upsampling_method: str = 'transposed',
#         dropout_p: float = 0,
#         skip_block: bool = True
#     ):
#         super(StackedDecoderBlock, self).__init__()
#         self.in_channels = in_channels
#         if out_channels is not None:
#             self.out_channels = out_channels
#         else:
#             self.out_channels = in_channels * 2
#         self.kernel_size = kernel_size
#         self.upsampling_method = upsampling_method
#         self.dropout_p = dropout_p
#         self.skip_block = skip_block
#         self.layer = layers
#
#         stack = []
#
#         # Define upsampling method.
#         if upsampling_method == 'transposed':
#             self.convT = DecoderBlock(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=2,
#                 padding=2,
#                 dropout_p=dropout_p,
#                 skip_block=skip_block
#             )
#         # TODO: Implement down and decimate methods.
#         elif upsampling_method == 'upsample':
#             pass
#         elif upsampling_method == 'unmaxpool':
#             pass
#
#         # Define convolutional layers.
#         for i in range(layers - 1):
#             stack.append(
#                 EncoderBlock(
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     stride=1,
#                     padding='same',
#                     leak=0
#                 )
#             )
#
#         # Register layers.
#         self.conv_stack = nn.Sequential(*stack)
#
#     def forward(self, data: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
#         """Forward method.
#
#         Args:
#             data (tensor): Input feature map.
#             output_size (torch.Size): Output size of this layer's output.
#         Returns:
#             (tensor): Output feature map.
#         """
#         x = self.convT(data, output_size=output_size)
#         output = self.conv_stack(x)
#         return output
#
#
#

# # Define convolutional layers.
# for i in range(layers - 1):
#     c_in = in_channels if i == 0 else out_channels
#     stack.append(
#         EncoderBlock(
#             in_channels=c_in,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             padding='same',
#             leak=leak
#         )
#     )
#
# # Define downsampling method.
# if downsampling_method == 'conv':
#     stack.append(
#         EncoderBlock(
#             in_channels=out_channels if layers > 1 else in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=2,
#             padding=2,
#             leak=leak
#         )
#     )
# # TODO: Implement down and decimate methods.
# elif downsampling_method == 'down':
#     pass
# elif downsampling_method == 'decimate':
#     pass
# elif downsampling_method == 'maxpool':
#     stack.append(nn.MaxPool2d(2))
#
# Register layers.
# self.conv_stack = nn.Sequential(*stack)


