import torch
import torch.nn as nn


from typing import Optional, Union, List


class EncoderBlock(nn.Module):
    _activations = {
        'relu': lambda leak: nn.ReLU() if not leak else nn.LeakyReLU(leak),
        'sigmoid': lambda _: nn.Sigmoid(),
        'tanh': lambda _: nn.Tanh(),
        'glu': lambda _: nn.GLU(),
    }
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
        leak: float = 0.2,
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
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if activation_fn:
            self.activation = self._activations[activation_fn](leak)
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
        output = self.activation(x)
        return output


ex_scheme = [
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
     }
]


class StackedEncoderBlock(nn.Module):
    """Stacks multiple encoder blocks together.

    Useful for constructing larger and more complex encoder blockers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int or None): Number of output channels. If None,
            defaults to 2 * in_channels.
        scheme (list): Multi-block encoder scheme. Default: None.
        # layers (int): Number of encoder blocks to stack.
        # kernel_size (int or tuple): Size or shape of the convolutional filter.
        #     Default: 5.
        # downsampling_method (str): Type of downsampling. Default: 'conv'.
        # leak (float): Negative slope value for leaky ReLU. Default: 0.2.
    """
    def __init__(
        self,
        layers: int = 1,
        scheme: Optional[List] = None,
    ):
        super(StackedEncoderBlock, self).__init__()

        # If scheme is passed in, we ignore layers arguments. Otherwise,
        # scheme must be a 1 element list
        #
        # if scheme:
        #     kernel_sizes = scheme['kernel_size']
        #     activations = scheme['activation']
        #     assert len(kernel_sizes) == len(activations) == layers, \
        #         f"Encoding scheme must match the number of layers."
        #
        # self.in_channels = in_channels
        # if out_channels is not None:
        #     self.out_channels = out_channels
        # else:
        #     self.out_channels = in_channels * 2
        # self.kernel_size = kernel_size
        # self.downsampling_method = downsampling_method
        # self.leak = leak
        # self.layer = layers
        #
        # stack = []
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

        # Register layers.
        # self.conv_stack = nn.Sequential(*stack)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
        Returns:
            (tensor): Output feature map.
        """
        output = self.conv_stack(data)
        return output


class DecoderBlock(nn.Module):
    """Fully-customizable decoder block layer for base separation models.

    By default, this block doubles the spatial dimensions (freq and time) but
    halves the feature dimension (channels). Corresponds with the encoder at
    the same depth level in the encoder/decoder architecture.

    Args:
    in_channels (int): Number of input channels.
    out_channels (int or None): Number of output channels. If None, defaults
        to 2 * in_channels.
    kernel_size (int or tuple): Size or shape of the convolutional filter.
        Default: 5.
    stride (int or tuple): Size or shape of the stride. Default: 2.
    padding (int or tuple): Size of zero-padding added to each side.
        Default: 2.
    dropout (float): Dropout probability. Default: 0.
    skip_block (bool): True if block has a skip connection, false otherwise.
        Default: True.
"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_size: Union[int, tuple] = 5,
            stride: Union[int, tuple] = 2,
            padding: Union[int, str] = 2,
            dropout: float = 0,
            skip_block: bool = True,
    ):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels * 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_p = dropout
        self.skip_block = skip_block

        # Define decoder block sequence.
        # tranposedconv -> batchnorm -> relu -> dropout
        self.convT = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        )

        # Define block layers.
        self.bn = nn.BatchNorm2d(out_channels) if skip_block else nn.Identity()
        self.relu = nn.ReLU() if skip_block else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, data: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
            output_size (torch.Size): Output size of this layer's output.
        Returns:
            (tensor): Output feature map.
        """
        x = self.convT(data, output_size=output_size)
        x = self.bn(x)
        x = self.relu(x)
        output = self.dropout(x)
        return output


class StackedDecoderBlock(nn.Module):
    """Stacks multiple denoder blocks together.

    Useful for constructing larger and more complex decoder blockers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int or None): Number of output channels. If None,
            defaults to 2 * in_channels.
        layers (int): Number of decoder blocks to stack.
        kernel_size (int or tuple): Size or shape of the convolutional filter.
            Default: 5.
        upsampling_method (str): Type of upsampling. Default: 'transposed'.
        dropout (float): Dropout probability. Default: 0.
        skip_block (bool): True if block has a skip connection, false otherwise.
            Default: True.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            layers=1,
            scheme: Optional[dict] = None,
            kernel_size: Union[int, tuple] = 5,
            upsampling_method: str = 'transposed',
            dropout: float = 0,
            skip_block: bool = True
    ):
        super(StackedDecoderBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels * 2
        self.kernel_size = kernel_size
        self.upsampling_method = upsampling_method
        self.dropout = dropout
        self.skip_block = skip_block
        self.layer = layers

        stack = []

        # Define upsampling method.
        if upsampling_method == 'transposed':
            self.convT = DecoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=2,
                dropout=dropout,
                skip_block=skip_block
            )
        # TODO: Implement down and decimate methods.
        elif upsampling_method == 'upsample':
            pass
        elif upsampling_method == 'unmaxpool':
            pass

        # Define convolutional layers.
        for i in range(layers - 1):
            stack.append(
                EncoderBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding='same',
                    leak=0
                )
            )

        # Register layers.
        self.conv_stack = nn.Sequential(*stack)

    def forward(self, data: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input feature map.
            output_size (torch.Size): Output size of this layer's output.
        Returns:
            (tensor): Output feature map.
        """
        x = self.convT(data, output_size=output_size)
        output = self.conv_stack(x)
        return output
