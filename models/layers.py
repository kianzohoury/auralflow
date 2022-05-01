import inspect

import torch
import torch.nn as nn
import math
import torch.nn.functional as functional

from typing import Optional, Union, Tuple


_ACTIVATIONS = {
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "elu",
    "prelu",
    "glu",
    "selu",
}
_DOWN_LAYERS = {
    "max_pool",
    "avg_pool",
    "conv",
    "downsample",
}
_UP_LAYERS = {
    "transpose",
    "nearest",
    "linear",
    "bilinear",
    "trilinear",
    "cubic",
}
_BLOCK_TYPES = {"conv", "soft_conv", "recurrent", "upsampler", "downsampler"}


def _get_conv_output_size(
    h_in: int, w_in: int, stride: int, kernel_size: int
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
    h_in: int, w_in: int, h_out: int, w_out: int, stride: int, kernel_size: int
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
    h_in: int, w_in: int, h_out: int, w_out: int, kernel_size: int
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
    activation_fn: str, param: Union[int, float] = None
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
        if activation_fn == "relu":
            return nn.ReLU()
        elif activation_fn == "leaky_relu":
            return nn.LeakyReLU(negative_slope=param)
        elif activation_fn == "elu":
            return nn.ELU(alpha=param)
        elif activation_fn == "prelu":
            return nn.PReLU(num_parameters=param)
        elif activation_fn == "glu":
            return nn.GLU()
        elif activation_fn == "sigmoid":
            return nn.Sigmoid()
        elif activation_fn == "tanh":
            return nn.Tanh()
    raise ValueError(
        f"Activation function must be one of {_ACTIVATIONS},"
        f" but received {activation_fn}."
    )


class _AEBlock(nn.Module):
    """Base and factory class used to generate all autoencoder blocks.

    This class is not meant to be instantiated.
    """

    @staticmethod
    def generate_block(
        block_type: str, dim: int, **kwargs
    ) -> Optional["_AEBlock"]:
        """Instantiates an autoencoder block.

        Args:
            block_type (str): Block type to request.
            dim (int): Number of spatial dimensions.

        Returns:
            (optional[_AEBlock]): An autoencoder block.

        Raises:
            ValueError: Raised if the block specifications are invalid.
        """
        if block_type not in _BLOCK_TYPES:
            raise ValueError(
                f"{block_type} is not a valid autoencoder block type."
            )
        elif dim != 1 and dim != 2:
            raise ValueError(
                "Spatial dimensions of autoencoder block must be 1d or 2d,"
                f" but received {dim}."
            )
        elif block_type == "conv":
            return ConvBlock(num_dims=dim, **kwargs)
        elif block_type == "soft_conv":
            return SoftConv(num_dims=dim, **kwargs)
        elif block_type == "recurrent":
            return _RecurrentBlock.generate_recurrent_block(
                num_dims=dim, **kwargs
            )
        elif block_type == "downsampler" and dim == 1:
            return _DownBlock1d(**kwargs)
        elif block_type == "downsampler" and dim == 2:
            return _DownBlock2d(**kwargs)
        elif block_type == "upsampler" and dim == 1:
            return _UpBlock1d(**kwargs)
        elif block_type == "upsampler" and dim == 2:
            return _UpBlock2d(**kwargs)
        return None


class ConvBlock(_AEBlock):
    """Convolutional block.

    Args:
        num_dims (int): Number of spatial dimensions. If audio is in time-freq
            domain, num_dims = 2, otherwise num_dims = 1.
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
        num_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[tuple, str, int] = "same",
        activation_fn: str = "relu",
        batch_norm: bool = True,
        use_bias: bool = True,
        activation_param: Optional[Union[int, float]] = None,
        dropout_p: float = 0,
    ):
        super(ConvBlock, self).__init__()

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

        conv = nn.Conv1d if num_dims == 1 else nn.Conv2d
        bn = nn.BatchNorm1d if num_dims == 1 else nn.BatchNorm2d
        dropout = nn.Dropout if num_dims == 1 else nn.Dropout2d

        layers = [
            conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.use_bias,
            )
        ]

        layers.extend([bn(self.out_channels)] if batch_norm else [])
        layers.append(_get_activation(activation_fn, activation_param))
        layers.extend([dropout(dropout_p)] if dropout_p > 0 else [])
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        return self.layers(data)


class _DownBlock2d(_AEBlock):
    """2-dimensional downsampling autoencoder block.

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
        scale_factor (optional[float]): Scaling factor for 'downsampling'
            via interpolation.

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
        scale_factor: Optional[float] = 0,
    ):
        super(_DownBlock2d, self).__init__()

        if down_method not in _DOWN_LAYERS:
            raise ValueError(
                f"Downsampling layer must be one of {_DOWN_LAYERS}, but"
                f" received {down_method}."
            )
        elif down_method == "conv":
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

        if down_method == "max_pool":
            self.down = nn.MaxPool2d(self.kernel_size, stride=stride)
        elif down_method == "avg_pool":
            self.down = nn.AvgPool2d(self.kernel_size, stride=stride)
        elif down_method == "downsample":
            self.down = DownSample(scale_factor=scale_factor)
        elif down_method == "conv":
            padding = _get_conv_padding(
                h_in=n_bins,
                w_in=n_samples,
                h_out=n_bins // 2,
                w_out=n_samples // 2,
                kernel_size=kernel_size,
            )
            self.down = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """"""
        return self.down(data)


class _DownBlock1d(_AEBlock):
    """1-dimensional downsampling autoencoder block.

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
        scale_factor: Optional[float] = 0,
    ):
        super(_DownBlock1d, self).__init__()

        if down_method not in _DOWN_LAYERS:
            raise ValueError(
                f"Downsampling layer must be one of {_DOWN_LAYERS},"
                f" but received {down_method}."
            )
        elif down_method == "conv":
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

        if down_method == "max_pool":
            self.down = nn.MaxPool1d(self.kernel_size, stride=stride)
        elif down_method == "avg_pool":
            self.down = nn.AvgPool1d(self.kernel_size, stride=stride)
        elif down_method == "downsample":
            self.down = DownSample(scale_factor=scale_factor)
        elif down_method == "conv":
            padding, _ = _get_conv_padding(
                h_in=0,
                w_in=n_samples,
                h_out=0,
                w_out=n_samples // 2,
                kernel_size=kernel_size,
            )
            self.down = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.down = nn.Identity()

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        return self.down(data)


class _UpBlock2d(_AEBlock):
    """2-dimensional upsampling autoencoder block.

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
        elif up_method != "transpose" and scale_factor is None:
            raise ValueError(
                f"Must specify scale_factor for non-transpose upsampling"
                f" layer, but received {up_method}."
            )
        elif up_method == "transpose":
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

        self.up_method = up_method
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_bins_in = n_bins_in
        self.n_samples_in = n_samples_in
        self.in_channels = in_channels
        self.out_channels = out_channels

        if up_method == "transpose":
            padding = _get_transpose_padding(
                h_in=n_bins_in,
                w_in=n_samples_in,
                h_out=n_bins_out,
                w_out=n_samples_out,
                stride=stride,
                kernel_size=kernel_size,
            )
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
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
                    padding="same" if same_padding else 0,
                ),
            )
            self.is_transpose = False

    def forward(
        self, data: torch.FloatTensor, output_size: torch.Size
    ) -> torch.FloatTensor:
        """Forward method."""
        if self.is_transpose:
            output = self.up(data, output_size=output_size)
        else:
            output = self.up(data)
        return output


class _UpBlock1d(_AEBlock):
    """1-dimensional upsampling autoencoder block.

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
        elif up_method != "transpose" and scale_factor is None:
            raise ValueError(
                f"Must specify scale_factor for non-transpose upsampling"
                f" layer, but received {up_method}."
            )
        elif up_method == "transpose":
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

        self.up_method = up_method
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_samples_in = n_samples_in
        self.in_channels = in_channels
        self.out_channels = out_channels

        if up_method == "transpose":
            _, padding = _get_transpose_padding(
                h_in=0,
                w_in=n_samples_out,
                h_out=0,
                w_out=n_samples_out,
                stride=stride,
                kernel_size=kernel_size,
            )
            self.up = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
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
                    padding="same" if same_padding else 0,
                ),
            )
            self.is_transpose = False

    def forward(
        self, data: torch.FloatTensor, output_size: torch.Size
    ) -> torch.FloatTensor:
        """Forward method."""
        if self.is_transpose:
            output = self.up(data, output_size=output_size)
        else:
            output = self.up(data)
        return output


class SoftConv(_AEBlock):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_targets: int,
        num_dims: int,
    ):
        super(SoftConv, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_targets = num_targets
        self.num_dims = num_dims
        conv = nn.Conv1d if num_dims == 1 else nn.Conv2d

        self.soft_conv = conv(
            in_channels=hidden_channels,
            out_channels=num_targets,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.separator_convs = nn.ModuleList()
        for _ in range(num_targets):
            self.separator_convs.append(
                conv(
                    in_channels=num_targets,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                )
            )

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        data = self.soft_conv(data)
        output = []
        for i in range(self.num_targets):
            output.append(self.separator_convs[i](data))
        output = torch.stack(output, dim=-1).float()
        return output


class _RecurrentBlock(_AEBlock):
    """Factory class for recurrent blocks."""

    @staticmethod
    def generate_recurrent_block(num_dims: int, **kwargs) -> "_RecurrentBlock":
        if num_dims == 1:
            return RecurrentBlock1d(**kwargs)
        else:
            return RecurrentBlock2d(**kwargs)


class RecurrentBlock2d(_RecurrentBlock):
    def __init__(self, **kwargs):
        super(RecurrentBlock2d, self).__init__()


class RecurrentBlock1d(_RecurrentBlock):
    def __init__(self, **kwargs):
        super(RecurrentBlock1d, self).__init__()


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
        assert len(data.shape) == 4, (
            f"Shape of input must be 3-dimensional, "
            f"but received input of size {len(data.shape) - 1}."
        )
        n, c, w, h = data.shape
        data = data.flatten(start_dim=-2)
        data = self.linear(data)
        out = self.glu(data)
        out = out.reshape((n, c, w // 2, h // 2))
        return out
