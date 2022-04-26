
import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import List, Optional, Tuple, Union


LAYER_TYPES = {
    'conv',
    'transpose_conv',
    'max_pool',
    'upsample',
    'downsample',
    'batch_norm',
    'dropout',
    'leaky_relu',
    'relu',
    'sigmoid',
    'tanh'
}


class DownSample2D(nn.Module):
    """Downsampling layer module implemented via interpolation."""
    def __init__(self, input_size: Tuple, scale_factor: int):
        super(DownSample2D, self).__init__()
        self._input_size = input_size
        self._scale_factor = scale_factor

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input data.

        Returns:
            (tensor): Downsampled data.
        """
        output = functional.interpolate(
            input=data,
            size=self._input_size,
            scale_factor=self._scale_factor
        )
        return output

class TransposeConv2D(nn.Module):
    """Wrapper class for nn.ConvTranspose2d."""
    def __init__(self, layers: List[nn.Module]):
        super(TransposeConv2D, self).__init__()
        assert len(layers) > 0, "Layers cannot be empty."
        assert isinstance(layers[0], nn.ConvTranspose2d),\
            "First layer must be a transpose convolution."
        self.transpose = layers.pop(0)
        self.layers = nn.Sequential(*layers)

    def forward(self,
                data: torch.Tensor,
                output_size: torch.Size) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Input data.
            output_size (size): Output size of transpose conv operation.

        Returns:
            (tensor): Output data.
        """
        data = self.transpose(data, output_size=output_size)
        output = self.layers(data)
        return output


class AdaptiveLayerNode(object):
    """Underlying datastructure used for parsing and constructing layers.

    Args:
        layer_type (str or None): The layer type. Default: None.
        block_type (str): The block type. Default: 'encoder'.
        param (int or float or None): The layer's parameter. Default: None.
        next_layer (AdaptiveLayerNode): The next layer within the block.
    """
    def __init__(self,
                 layer_type: Optional[str] = None,
                 block_type: str = 'encoder',
                 param: Optional[Union[int, float]] = None,
                 next_layer: 'AdaptiveLayerNode' = None):

        super(AdaptiveLayerNode, self).__init__()
        assert layer_type is None or layer_type in LAYER_TYPES, \
            f"Unknown layer {layer_type} was passed in."
        self.layer_type = layer_type
        self.block_type = block_type
        self.next_layer = next_layer
        if layer_type in {
            'conv',
            'transpose_conv',
            'max_pool',
            'upsample',
            'downsample'
        }:
            assert isinstance(param, int), \
                (f"Value for layer {layer_type} must be an int, but received"
                 f" a value of type {type(param)}.")
        elif layer_type in {'dropout', 'leaky_relu'}:
            assert isinstance(param, float), \
                (f"Value for layer {layer_type} must be a float, but received"
                 f"a value of type {type(param)}.")
        self.param = param
        if block_type == 'encoder':
            self.downsampling_head = False
        elif block_type == 'decoder':
            self.upsampling_head = False
            self.split_point = False

    def __repr__(self):
        return f"<Type: {self.layer_type}, Block: {self.block_type}>"


class StackedEncoderBlock(nn.Module):
    """Encoder block module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channnels.
        h_in (int): Input height.
        w_in (int): Input width.
        h_out (int): Target height.
        w_out (int): Target width.
        block_scheme (AdaptiveLayerNode): Encoder block scheme.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 h_in: int,
                 w_in: int,
                 h_out: int,
                 w_out: int,
                 block_scheme: AdaptiveLayerNode):
        super(StackedEncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._h_in = h_in
        self._w_in = w_in
        self._h_out = h_out
        self._w_out = w_out
        encoding_layers = []
        downsampling_layers = []

        encoding_head = AdaptiveLayerNode(
            block_type='encoder',
            next_layer=block_scheme
        )

        down_phase = False
        while encoding_head:
            if encoding_head.downsampling_head:
                down_phase = True
            if encoding_head.layer_type == 'conv':
                if down_phase:
                    padding = get_conv_padding(
                        h_in=h_in,
                        w_in=w_in,
                        h_out=h_out,
                        w_out=w_out,
                        kernel_size=encoding_head.param
                    )
                    stride = 2
                else:
                    padding = 'same'
                    stride = 1
                layer_module = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=encoding_head.param,
                    stride=stride,
                    padding=padding
                )
                in_channels = out_channels
            elif encoding_head.layer_type == 'max_pool':
                layer_module = nn.MaxPool2d(
                    kernel_size=encoding_head.param
                )
            elif encoding_head.layer_type == 'downsample':
                layer_module = DownSample2D(
                    input_size=(h_out, w_out),
                    scale_factor=encoding_head.param
                )
            elif encoding_head.layer_type in LAYER_TYPES:
                if encoding_head.layer_type == 'batch_norm':
                    param = out_channels
                else:
                    param = encoding_head.param
                layer_module = get_layer(
                    layer_type=encoding_head.layer_type,
                    param=param
                )
            else:
                layer_module = None
            if down_phase and layer_module is not None:
                downsampling_layers.append(layer_module)
            elif layer_module is not None:
                encoding_layers.append(layer_module)
            encoding_head = encoding_head.next_layer

        self.encoder = nn.Sequential(*encoding_layers)
        self.down = nn.Sequential(*downsampling_layers)

    def forward(self, data: torch.Tensor) -> Tuple:
        """Forward method.

        Args:
            data (tensor): Input data.

        Returns:
            (tuple): A tuple of the output and intermediate skip data.
        """
        skip_data = self.encoder(data)
        output = self.down(skip_data)
        return output, skip_data

    def is_single_block(self) -> bool:
        """Returns True if more than one convolutional layer is used."""
        return not len(list(self.encoder.children()))

    def pop_layer(self) -> None:
        """Removes the last downsampling layer."""
        layers = list(self.down.children())
        if len(layers) > 0:
            self.down = nn.Sequential(*layers[:-1])


class StackedDecoderBlock(nn.Module):
    """Encoder block module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channnels.
        h_in (int): Input height.
        w_in (int): Input width.
        h_out (int): Target height.
        w_out (int): Target width.
        block_scheme (AdaptiveLayerNode): Decoder block scheme.
        use_skip (bool): Whether to use skip connections.
        use_dropout (bool): Whether to use dropout.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 h_in: int,
                 w_in: int,
                 h_out: int,
                 w_out: int,
                 block_scheme: AdaptiveLayerNode,
                 use_skip: bool,
                 use_dropout: bool):
        super(StackedDecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._h_in = h_in
        self._w_in = w_in
        self._h_out = h_out
        self._w_out = w_out
        self._use_skip = use_skip
        self._use_dropout = use_dropout
        self._use_transpose = False
        decoding_layers = []
        upsampling_layers = []

        decoding_head = AdaptiveLayerNode(
            block_type='decoder',
            next_layer=block_scheme
        )

        has_decoder_layers = False
        ptr = decoding_head
        while ptr.next_layer:
            if ptr.next_layer.layer_type == 'transpose_conv':
                self._use_transpose = True
            elif ptr.next_layer.layer_type == 'conv':
                has_decoder_layers = True
                break
            ptr = ptr.next_layer

        up_phase = True
        while decoding_head:
            if decoding_head.split_point:
                up_phase = False
                if use_skip:
                    in_channels *= 2
            if decoding_head.layer_type == 'transpose_conv':
                if use_skip and not has_decoder_layers:
                    in_channels *= 2
                padding = get_transpose_padding(
                    h_in=h_in,
                    w_in=w_in,
                    h_out=h_out,
                    w_out=w_out,
                    stride=2,
                    kernel_size=decoding_head.param
                )
                layer_module = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=decoding_head.param,
                    stride=2,
                    padding=padding
                )
                in_channels = out_channels
            elif decoding_head.layer_type == 'conv':
                padding = 'same'
                stride = 1
                layer_module = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=decoding_head.param,
                    stride=stride,
                    padding=padding
                )
                in_channels = out_channels
            elif decoding_head.layer_type == 'upsample':
                layer_module = nn.Upsample(
                    size=torch.Size((h_in, w_in)),
                    scale_factor=decoding_head.param
                )
            elif decoding_head.layer_type in LAYER_TYPES:
                if decoding_head.layer_type == 'batch_norm':
                    layer_module = get_layer(
                        layer_type=decoding_head.layer_type,
                        param=out_channels
                    )
                elif decoding_head.layer_type == 'dropout' and not use_dropout:
                    layer_module = None
                else:
                    layer_module = get_layer(
                        layer_type=decoding_head.layer_type,
                        param=decoding_head.param
                    )
            else:
                layer_module = None
            if up_phase and layer_module is not None:
                upsampling_layers.append(layer_module)
            elif layer_module is not None:
                decoding_layers.append(layer_module)
            decoding_head = decoding_head.next_layer

        self.decoder = nn.Sequential(*decoding_layers)
        if self._use_transpose:
            self.up = TransposeConv2D(upsampling_layers)
        else:
            self.up = nn.Sequential(*upsampling_layers)

    def forward(self,
                data: torch.Tensor,
                skip_data: Optional[torch.Tensor] = None,
                output_size: Optional[torch.Size] = None) -> torch.Tensor:
        """Forward method.

        Args:
            data (tensor): Direct input data from previous layer.
            skip_data (tensor or None): Data from skip connection.
            output_size (size or None): Output size for transpose convolutions.

        Returns:
            (tuple): A tuple of the output and intermediate skip data.
        """
        if self._use_transpose and output_size is not None:
            data = self.up(data, output_size=output_size)
        else:
            data = self.up(data)
        if self._use_skip and skip_data is not None:
            data = torch.cat([data, skip_data], dim=1)
            output = self.decoder(data)
        else:
            output = self.decoder(data)
        return output

    def is_single_block(self):
        """Returns True if more than one convolutional layer is used."""
        return not len(list(self.decoder.children()))





        # while downsampling_head:
        #     if downsampling_head.layer_type == 'conv':
        #         padding = get_conv_padding(
        #             h_in=h_in,
        #             w_in=w_in,
        #             h_out=h_out,
        #             w_out=w_out,
        #             kernel_size=downsampling_head.param
        #         )
        #         layer_module = nn.Conv2d(
        #             in_channels=in_channels,
        #             out_channels=out_channels,
        #             kernel_size=downsampling_head.param,
        #             stride=2,
        #             padding=padding
        #         )
        #     elif downsampling_head.layer_type == 'max_pool':
        #         layer_module = nn.MaxPool2d(
        #             kernel_size=downsampling_head.param
        #         )
        #     elif downsampling_head.layer_type == 'downsample':
        #         layer_module = DownSample2D(
        #             input_size=(h_out, w_out),
        #             scale_factor=downsampling_head.param
        #         )
        #     else:
        #         layer_module = get_layer(
        #             downsampling_head.layer_type,
        #             downsampling_head.param
        #         )
        # encoding_layers.append(layer_module)
        # encoding_head = encoding_head.next_layer
        #
        #     if layer_type == 'conv':
        #         kernel_size = layer['param']
        #         if is_encoder:
        #             if layer.get('down', False):
        #                 padding = get_conv_padding(
        #                     h_in=h_in,
        #                     w_in=w_in,
        #                     h_out=h_out,
        #                     w_out=w_out,
        #                     kernel_size=kernel_size
        #                 )
        #                 stride = 2
        #             else:
        #                 padding = 'same'
        #                 stride = 1
        #         else:
        #             if skip_connections and len(conv_stack) == 0 and not skip_last:
        #                 in_channels = in_channels * 2
        #
        #             padding = 'same'
        #             stride = 1
        #
        #         layer_module = nn.Conv2d(
        #             in_channels=in_channels,
        #             out_channels=out_channels,
        #             kernel_size=kernel_size,
        #             stride=stride,
        #             padding=padding
        #         )
        #
        #         in_channels = out_channels
        #
        #     elif layer_type == 'transpose_conv':
        #         kernel_size = layer['param']
        #         padding = get_transpose_padding(
        #             h_in=h_in,
        #             w_in=w_in,
        #             h_out=h_out,
        #             w_out=w_out,
        #             stride=2,
        #             kernel_size=kernel_size
        #         )
        #         print(h_in, w_in, h_out, w_out, padding)
        #         print(in_channels)
        #         if skip_connections and not first_decoder and len(scheme['conv_stack']) == 0:
        #             in_channels *= 2
        #
        #         layer_module = nn.ConvTranspose2d(
        #             in_channels=in_channels,
        #             out_channels=out_channels,
        #             kernel_size=kernel_size,
        #             stride=2,
        #             padding=padding
        #         )
        #
        #         in_channels = out_channels
        #     elif layer_type == 'upsample':
        #         layer_module = nn.Upsample(
        #             size=torch.Size((h_in, w_in)),
        #             scale_factor=2
        #         )
        #     elif layer_type == 'max_pool':
        #         kernel_size = layer['param']
        #         layer_module = nn.MaxPool2d(kernel_size)
        #     elif layer_type == 'batch_norm':
        #         layer_module = nn.BatchNorm2d(out_channels)
        #     elif layer_type == 'dropout' and use_dropout:
        #         dropout_p = layer['param']
        #         layer_module = nn.Dropout2d(dropout_p)
        #     elif layer_type in {'relu', 'leaky_relu', 'sigmoid', 'tanh'}:
        #         layer_module = get_activation(layer_type, layer['param'])
        #     else:
        #         layer_module = nn.Identity()
        #
        #
        #     if is_encoder:
        #         print(i, 123, layer_type, split_index)
        #         if i >= split_index:
        #             down.append(layer_module)
        #         else:
        #             conv_stack.append(layer_module)
        #     else:
        #         if i < split_index:
        #             up.append(layer_module)
        #         else:
        #             conv_stack.append(layer_module)
        #
        #
        #
        # self.conv_stack = nn.Sequential(*conv_stack)
        #
        # if is_encoder:
        #     self.down = DownSampler(down)
        # else:
        #     self.deconv_pre = up[0]
        #     self.deconv_post = nn.Sequential(*up[1:])


def get_transpose_padding(h_in: int,
                          w_in: int,
                          h_out: int,
                          w_out: int,
                          stride: int,
                          kernel_size: int) -> Tuple:
    """Gets the padding needed for a specific shape after a transpose layer.

    Args:
        h_in (int): Input height.
        w_in (int): Input width.
        h_out (int): Target height.
        w_out (int): Target width.
        stride (int): Stride.
        kernel_size (int). Kernel size.

    Returns:
        (tuple): The output padding.

    """
    h_pad = math.ceil((kernel_size - h_out + stride * (h_in - 1)) / 2)
    w_pad = math.ceil((kernel_size - w_out + stride * (w_in - 1)) / 2)
    return h_pad, w_pad


def get_conv_padding(h_in: int,
                     w_in: int,
                     h_out: int,
                     w_out: int,
                     kernel_size: int) -> Tuple:
    """Gets the padding needed for a specific shape after a conv layer.

    Args:
        h_in (int): Input height.
        w_in (int): Input width.
        h_out (int): Target height.
        w_out (int): Target width.
        kernel_size (int). Kernel size.

    Returns:
        (tuple): The output padding.
    """
    h_pad = max(0, math.ceil((2 * h_out - 2 + kernel_size - h_in) / 2))
    w_pad = max(0, math.ceil((2 * w_out - 2 + kernel_size - w_in) / 2))
    return h_pad, w_pad


def get_layer(layer_type: str,
              param: Optional[Union[int, float]] = None) -> nn.Module:
    """Helper method that returns a requested layer.

    Args:
        layer_type (str): Layer type.
        param (int or float or None): The layer's parameter.

    Returns:
        (nn.Module): The returned layer module.

    Raises:
        ValueError: Raised when the layer type is not a valid layer.
    """
    if layer_type == 'batch_norm':
        return nn.BatchNorm2d(param)
    elif layer_type == 'dropout':
        return nn.Dropout2d(param)
    elif layer_type == 'relu':
        return nn.ReLU()
    elif layer_type == 'leaky_relu':
        return nn.LeakyReLU(param)
    elif layer_type == 'sigmoid':
        return nn.Sigmoid()
    elif layer_type == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"{layer_type} is not a valid layer.")


def process_block(block_scheme: List,
                  block_type: str = 'encoder') -> AdaptiveLayerNode:
    """Processes and verifies an raw autoencoder block scheme.

    Args:
        block_scheme (list): The encoder/decoder scheme.
        block_type: The block type. Default: 'encoder'.

    Returns:
        (AdaptiveLayerNode): The processed block.
    """

    assert len(block_scheme) > 0, f"Must specify at least 1 layer."
    head = prev = AdaptiveLayerNode(block_type=block_type)

    # Append a placeholder value for non-parameterized layers.
    for layer in block_scheme:
        if len(layer) == 1:
            layer.append(None)
    # Create a node for each layer.
    for i, (layer_type, param) in enumerate(block_scheme):
        layer_node = AdaptiveLayerNode(
            layer_type=layer_type,
            block_type=block_type,
            param=param
        )
        prev.next_layer = layer_node
        prev = prev.next_layer

    use_max_pool = False
    use_downsample = False
    use_upsample = False
    use_conv = False
    use_transpose = False
    ptr = head.next_layer

    # Enforce that only one of the downsampling/upsampling method is used.
    while ptr:
        if ptr.layer_type == 'conv':
            use_conv = True
        if block_type == 'encoder':
            if ptr.layer_type == 'max_pool':
                use_max_pool = True
            elif ptr.layer_type == 'downsample':
                use_downsample = True
            assert ptr.layer_type not in {'transpose_conv', 'upsample'}, \
                f"Cannot use {ptr.layer_type} with an encoder block."
        else:
            if ptr.layer_type == 'transpose_conv':
                use_transpose = True
            elif ptr.layer_type == 'upsample':
                use_upsample = True
            assert ptr.layer_type not in {'max_pool', 'downsample'}, \
                f"Cannot use {ptr.layer_type} with a decoder block."
        ptr = ptr.next_layer

    # Verify that the block is a valid construction.
    valid_encoder = use_conv
    valid_decoder = (use_conv and use_upsample) or use_transpose
    if block_type == 'encoder':
        assert valid_encoder, (
            f"Encoder must specify at least 1 valid convolutional layer."
        )
        assert use_max_pool ^ use_downsample or not \
            (use_max_pool and use_downsample), \
            f"Cannot use max_pool and downsample simultaneously."
    else:
        assert valid_decoder, (
            f"Decoder must specify at least 1 valid de-convolutional layer."
        )
        assert use_upsample ^ use_transpose or not \
            (use_upsample and use_transpose), \
            f"Cannot use transpose_conv and upsample simultaneously."

    # Encoder block needs one of max_pool, stride conv and downsample at the
    # tail, and decoder block needs one of transpose and upsample in the front.
    ptr = head
    if block_type == 'encoder':
        if use_max_pool:
            last_layer = 'max_pool'
        elif use_downsample:
            last_layer = 'downsample'
        else:
            last_layer = 'conv'
        if last_layer == 'conv':
            num_convs = 0
            while ptr.next_layer:
                num_convs += 1 if ptr.next_layer.layer_type == 'conv' else 0
                ptr = ptr.next_layer
            ptr = head
            while ptr.next_layer:
                num_convs -= 1 if ptr.next_layer.layer_type == 'conv' else 0
                if num_convs == 0:
                    ptr.next_layer.downsampling_head = True
                    break
                ptr = ptr.next_layer
        elif last_layer == 'max_pool' or last_layer == 'downsample':
            while ptr.next_layer and ptr.next_layer.layer_type != last_layer:
                ptr = ptr.next_layer
            ptr.next_layer.downsampling_head = True
            if ptr.next_layer:
                temp = ptr.next_layer
                ptr.next_layer = ptr.next_layer.next_layer
                temp.next_layer = None
                while ptr.next_layer:
                    ptr = ptr.next_layer
                ptr.next_layer = temp
        head = head.next_layer
    else:
        first_layer = 'transpose_conv' if use_transpose else 'upsample'
        while ptr.next_layer and ptr.next_layer.layer_type != first_layer:
            ptr = ptr.next_layer
        ptr.next_layer.upsampling_head = True
        if ptr.next_layer:
            temp = ptr.next_layer
            ptr.next_layer = ptr.next_layer.next_layer
            temp.next_layer = None
            temp.next_layer = head.next_layer
            head = temp
        ptr = head
        # Mark the next conv layer to get stopping point for `up` layers.
        while ptr:
            if ptr.layer_type == 'conv':
                ptr.split_point = True
            ptr = ptr.next_layer
    return head
