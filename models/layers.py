
import math
import torch
import torch.nn as nn

from typing import Optional, Union, List, Tuple


def _get_activation(
    activation_fn: Union[str, None],
    param: Optional[Union[int, float]] = None
) -> nn.Module:
    """Helper method to return an activation function.

    Args:
        param (int or None): For parameterized activation functions.

    Returns:
        (nn.Module): An activation function.
    """
    if activation_fn == 'leaky_relu':
        return nn.LeakyReLU(param) if param is not None else nn.Identity()
    elif activation_fn == 'relu':
        return nn.ReLU()
    elif activation_fn == 'sigmoid':
        return nn.Sigmoid()
    elif activation_fn == 'tanh':
        return nn.Tanh()
    elif activation_fn == 'glu':
        return GLU2d(param) if param is not None else nn.Identity()
    else:
        return nn.Identity()


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
        padding (int or tuple): Size of zero-padding added to each side.
            Default: 2.
        activation_fn (str or None): Activation function. Default: 'relu'.
        batch_norm (bool): Whether to apply batch normalization. Default: True.
        bias (bool): Whether to include a bias term in the conv layer.
            Default: False.
        leak (float): Negative slope value if using leaky ReLU. Default: 0.2.
        num_glu_features (int): The number of features for GLU2d activation
            func. Default: None.
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
        num_glu_features: Optional[int] = None,
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
        self.leak = leak
        self.num_glu_features = num_glu_features
        self.maxpool = nn.MaxPool2d(2) if max_pool else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if activation_fn == 'leaky_relu':
            self.activation = _get_activation(activation_fn, leak)
        elif activation_fn == 'glu':
            self.activation = _get_activation(activation_fn,
                                              num_glu_features)
        else:
            self.activation = _get_activation(activation_fn)
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
        num_glu_features: Optional[int] = None
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
        if activation_fn == 'glu':
            self.activation = _get_activation(activation_fn,
                                              num_glu_features)
        else:
            self.activation = _get_activation(activation_fn)
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
        'bias',
    }

    def __init__(self, scheme: List[dict], block_type: str = 'encoder'):
        super(StackedBlock, self).__init__()
        assert len(scheme) > 0,\
            f"Must specify a non-empty block scheme, but received {scheme}."
        for layer_scheme in scheme:
            for key in layer_scheme.keys():
                assert key in self._keys, f"{key} is not a valid key."
        self.num_layers = len(scheme)
        self.block_type = block_type
        stack = []
        for layer_scheme in scheme:
            if block_type == 'encoder':
                stack.append(EncoderBlock(**layer_scheme))
            else:
                stack.append(DecoderBlock(**layer_scheme))
        # Register layers.
        self.layers_stack = nn.Sequential(*stack)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
        Returns:
            (tensor): Output feature map.
        """
        output = self.layers_stack(data)
        return output


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
