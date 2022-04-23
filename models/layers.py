
import torch
import torch.nn as nn
import math
from pprint import pprint
from typing import Optional, Union, List, Tuple


def get_transpose_padding(h_in: int, w_in: int, h_out: int, w_out: int,
                          stride: int, kernel_size: int) -> Tuple:
    """Computes the required transpose conv padding for a target shape."""
    h_pad = math.ceil((kernel_size - h_out + stride * (h_in - 1)) / 2)
    w_pad = math.ceil((kernel_size - w_out + stride * (w_in - 1)) / 2)
    return h_pad, w_pad


def get_conv_padding(h_in: int, w_in: int, h_out: int, w_out: int,
                     kernel_size: int) -> Tuple:
    """Computes the required conv padding."""
    h_pad = max(0, math.ceil((2 * h_out - 2 + kernel_size - h_in) / 2))
    w_pad = max(0, math.ceil((2 * w_out - 2 + kernel_size - w_in) / 2))
    return h_pad, w_pad






class EncoderBlock(nn.Module):
    """Fully-customizable encoder block layer for base separation models.

    By default, this block doubles the feature dimension (channels)
    but halves the spatial dimensions of the input (freq and time).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int or None): Number of output channels. If None,
            defaults to 2 * in_channels.
        kernel_size (int or tuple): Size or shape of the convolutional filter.
            Default: 5.
        stride (int or tuple): Size or shape of the stride. Default: 2.
        padding (int or str): Size of zero-padding added to each side.
            Default: 2.
        activation_fn (str or None): Activation function. Default: 'relu'.
        batch_norm (bool): Whether to apply batch normalization. Default: True.
        bias (bool): Whether to include a bias term in the conv layer.
            Default: False.
        leak (float): Negative slope value if using leaky ReLU. Default: 0.2.
        max_pool (bool): Whether to use maxpool. Default: False.
    Examples:
        >>> from models.layers import EncoderBlock
        >>> encoder = EncoderBlock(16, 32, 5, 2, 2, 'relu', batch_norm=True)
        >>> data = torch.rand((1, 16, 4, 1))
        >>> output = encoder(data)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Union[int, tuple] = 5,
        stride: Union[int, tuple] = 2,
        padding: Union[int, str] = 2,
        activation_fn: Optional[str] = 'relu',
        batch_norm: bool = True,
        bias: bool = False,
        leak: Optional[float] = None,
        max_pool: bool = False
    ):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels = in_channels * 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.leak = leak if leak is not None else 0
        self.maxpool = nn.MaxPool2d(2) if max_pool else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if activation_fn == 'leaky_relu' and self.leak > 0:
            self.activation = nn.LeakyReLU(leak)
        elif activation_fn == 'relu':
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
        Returns:
            (tensor): Output feature map.
        """
        x = self.conv(data)
        x = self.batchnorm(x)
        x = self.activation(x)
        output = self.maxpool(x)
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


class UNetNode(object):
    def __init__(self, layer_name: str = "",
                 parameter: Union[int, float] = 0,
                 next_layer=None):
        self.layer_name = layer_name
        self.parameter = parameter
        self.next_layer = next_layer

    def __str__(self):
        head = self
        result = []
        while head:
            result.append(f"<{head.layer_name}, {head.parameter or ''}>")
            head = head.next_layer
        return "\n".join(result)


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

    def __init__(self, in_channels: int, out_channels: int, num_bins: int,
                 num_samples: int, scheme: List[List],
                 block_type: str = 'encoder', skip_connections: bool = True):
        super(StackedBlock, self).__init__()
        assert len(scheme) > 0,\
            f"Must specify a non-empty block scheme, but received {scheme}."
        self.num_layers = len(scheme)
        self.block_type = block_type
        self.skip_connections = skip_connections
        print(block_type, in_channels, out_channels)
        spatial_resize_index = len(scheme) - 1 if block_type == 'encoder' else 0
        if block_type == 'encoder':
            while spatial_resize_index >= 0 and scheme[spatial_resize_index][0] not in ['conv', 'max_pool']:
                spatial_resize_index -= 1
        else:
            while spatial_resize_index < len(scheme) and scheme[spatial_resize_index][0] != 'conv':
                spatial_resize_index += 1


        encoder_stack = []
        decoder_stack = []
        down = []
        up = []

        for layer in scheme[:spatial_resize_index]:
            layer_name = layer[0]
            if layer_name == 'conv':
                kernel_size = layer[-1]
                if block_type == 'encoder':
                    encoder_stack.append(
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding='same')
                    )
                else:

                    up.append(
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding='same')
                    )
                in_channels = out_channels
            elif layer_name == 'transpose_conv':
                kernel_size = layer[-1]
                h_in, w_in = num_bins, num_samples
                h_out, w_out = num_bins * 2, num_samples * 2
                padding = get_transpose_padding(
                    h_in=h_in, w_in=w_in, h_out=h_out, w_out=w_out,
                    kernel_size=kernel_size, stride=2
                )
                up.append(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=kernel_size, stride=2,
                                       padding=padding
                    )
                )

            elif layer_name == 'batch_norm':
                if block_type == 'encoder':
                    encoder_stack.append(nn.BatchNorm2d(out_channels))
                else:
                    up.append(nn.BatchNorm2d(out_channels))
            elif layer_name == 'dropout':
                dropout_p = layer[-1]
                if block_type == 'encoder':
                    encoder_stack.append(nn.Dropout2d(dropout_p))
                else:
                    up.append(nn.Dropout2d(dropout_p))
            elif layer_name == 'leaky_relu':
                leak = layer[-1]
                if block_type == 'encoder':
                    encoder_stack.append(nn.LeakyReLU(leak))
                else:
                    up.append(nn.LeakyReLU(leak))
            elif layer_name == 'relu':
                if block_type == 'encoder':
                    encoder_stack.append(nn.ReLU())
                else:
                    up.append(nn.ReLU())
            elif layer_name == 'tanh':
                if block_type == 'encoder':
                    encoder_stack.append(nn.Tanh())
                else:
                    up.append(nn.Tanh())
            elif layer_name == 'sigmoid':
                if block_type == 'encoder':
                    encoder_stack.append(nn.Sigmoid())
                else:
                    up.append(nn.Sigmoid())
            else:
                if block_type == 'encoder':
                    encoder_stack.append(nn.Identity())
                else:
                    up.append(nn.Identity())

        for layer in scheme[spatial_resize_index:]:
            layer_name = layer[0]
            if layer_name == 'conv':
                kernel_size = layer[-1]
                if block_type == 'encoder':
                    h_in, w_in = num_bins, num_samples
                    h_out, w_out = h_in // 2, w_in // 2
                    padding = get_conv_padding(
                        h_in=h_in, w_in=w_in, h_out=h_out, w_out=w_out,
                        kernel_size=kernel_size
                    )
                    stride = 2
                    up.append(
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)
                    )
                else:
                    # if not decoder_stack:
                    #     in_channels = in_channels * (1 + int(skip_connections))
                    decoder_stack.append(
                            nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding='same')
                    )
                in_channels = out_channels
            elif layer_name == 'max_pool':
                down.append(nn.MaxPool2d(2))
            elif layer_name == 'batch_norm':
                if block_type == 'encoder':
                    down.append(nn.BatchNorm2d(out_channels))
                else:
                    decoder_stack.append(nn.BatchNorm2d(out_channels))
            elif layer_name == 'dropout':
                dropout_p = layer[-1]
                if block_type == 'encoder':
                    down.append(nn.Dropout2d(dropout_p))
                else:
                    decoder_stack.append(nn.Dropout2d(dropout_p))
            elif layer_name == 'leaky_relu':
                leak = layer[-1]
                if block_type == 'encoder':
                    down.append(nn.LeakyReLU(leak))
                else:
                    decoder_stack.append(nn.LeakyReLU(leak))
            elif layer_name == 'relu':
                if block_type == 'encoder':
                    down.append(nn.ReLU())
                else:
                    decoder_stack.append(nn.ReLU())
            elif layer_name == 'tanh':
                if block_type == 'encoder':
                    down.append(nn.Tanh())
                else:
                    decoder_stack.append(nn.Tanh())
            elif layer_name == 'sigmoid':
                if block_type == 'encoder':
                    down.append(nn.Sigmoid())
                else:
                    decoder_stack.append(nn.Sigmoid())
            else:
                if block_type == 'encoder':
                    down.append(nn.Identity())
                else:
                    decoder_stack.append(nn.Identity())

        if block_type == 'encoder':
            self.layers_stack = nn.Sequential(*encoder_stack)
            self.down = nn.Sequential(*down)
        else:
            self.layers_stack = nn.Sequential(*decoder_stack)
            self.up = nn.Sequential(*up)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
        Returns:
            (tensor): Output feature map.
        """
        output = self.layers_stack(data)
        return output


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

