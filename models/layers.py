
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as functional


from pprint import pprint
from collections import OrderedDict
from typing import Optional, Union, List, Tuple
from abc import ABCMeta, abstractmethod

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
    'nearest',
    'linear',
    'bilinear',
    'trilinear',
    'cubic'
}


def _get_conv_output_size(
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


def _get_transpose_padding(
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


def _get_conv_padding(
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


def _get_activation(
    activation_fn: str,
    param: Union[int, float] = None
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





#
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
#                  scheme: LayerNode,
#                  skip_connections: bool = True,
#                  last: bool = False,
#                  is_encoder: bool = True,
#                  first_decoder: bool = False,
#                  use_dropout: bool = True,
#                  skip_last: bool = True
#                  ):
#         super(StackedBlock, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         print(scheme)
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

# class DownSampler(nn.Module):
#     def __init__(self, layers):
#         super(DownSampler, self).__init__()
#         self.down = nn.Sequential(*layers)
#
#     def forward(self, data):
#         return self.down(data)

#
# class StackedEncoderBlock(StackedBlock):
#     def __init__(self, **kwargs):
#         super(StackedEncoderBlock, self).__init__(**kwargs)
#
#     def forward(self, data):
#         encoding = self.conv_stack(data)
#         output = self.down(encoding)
#         return output, encoding
#
#
# class StackedDecoderBlock(StackedBlock):
#     def __init__(self, **kwargs):
#         super(StackedDecoderBlock, self).__init__(**kwargs)

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


