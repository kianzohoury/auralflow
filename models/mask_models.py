import torch
import torch.nn as nn

from typing import Optional, Union, Tuple
from .base import SeparationModel
from .modules import AutoEncoder2d
from .layers import _get_activation
from transforms import _make_hann_window


class TFMaskUNet(nn.Module):
    """U-Net source separation model in the time-frequency domain.

    Uses the standard `soft-masking` technique to separate a single
    constituent audio source from its input mixture. The architecture
    implements a vanilla U-Net design, which involves a basic
    encoder-decoder scheme (without an additional bottleneck layer).
    The separation procedure is as follows:

    * The encoder first compresses an audio sample x in the time-frequency
      domain to a low-resolution representation.
    * The decoder receives the encoder's output as input, and reconstructs
      a new sample x~, which matches the dimensionality of x.
    * A an activation layer normalizes x~ to force its values to be between
      [0, 1], creating a `soft-mask`.
    * The mask is applied to the original audio sample x as an element-wise
      product, yielding the target source estimate y.

    Args:
        num_fft_bins (int): Number of STFT bins (otherwise known as filter
            banks). Note that only num_fft_bins // 2 + 1 are used due to the
            symmetry property of the fourier transform.
        num_samples (int): Number of samples (temporal dimension).
        num_channels (int): Number of audio channels.
        hop_length (int): Hop length.
        window_size (int): Window size.
        window_type (optional[str]): Windowing function for the stft transform.
            Default: 'hann'.
        trainable_window (bool): Whether to learn the windowing function.
            Default: False.
        max_depth (int): Maximum depth of the autoencoder. Default: 6.
        hidden_size (int): Initial hidden size of the autoencoder.
            Default: 16.
        kernel_size (union[Tuple, int]): Kernel sizes of the autoencoder. A
            tuple of (encoder_k_size, downsampler_k_size, decoder_k_size,
            upsampler_k_size) may be passed in. Otherwise, all kernels will
            share the same size. Default: 3.
        block_size (int): Depth of each encoder/decoder block. Default: 3.
        downsampler (str): Downsampling method employed by the encoder.
            Default: 'max_pool'.
        upsampler (str): Upsampling method employed by the decoder.
            Default: 'transpose".
        batch_norm (bool): Whether to use batch normalization. Default: True.
        layer_activation_fn (str): Activation function used for each
            autoencoder layer. Default: 'relu'.
        mask_activation_fn (str): Final activation used to construct the
            multiplicative soft-mask. Default: 'relu'.
        dropout_p (float): Dropout probability. If p > 0, dropout is used.
            Default: 0.
        use_skip (bool): Whether to concatenate skipped data from the encoder
            to the decoder. Default: True.
        normalize_input (bool): Whether to normalize the input. Note, that
            this layer simply uses batch norm instead of actual data whitening.
            Default: False.
    """

    def __init__(
        self,
        num_fft_bins: int,
        num_samples: int,
        num_channels: int,
        hop_length: int,
        window_size: int,
        max_depth: int = 6,
        hidden_size: int = 16,
        kernel_size: Union[Tuple, int] = 2,
        block_size: int = 3,
        downsampler: str = "max_pool",
        upsampler: str = "transpose",
        batch_norm: bool = True,
        layer_activation_fn: str = "relu",
        mask_activation_fn: str = "relu",
        dropout_p: float = 0,
        use_skip: bool = True,
        normalize_input: bool = False,
        window_type: Optional[str] = "hann",
    ):
        self.num_targets = 1
        self.num_fft = num_fft_bins
        self.num_bins = num_fft_bins // 2 + 1
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.hop_length = hop_length
        self.window_size = window_size
        self.max_depth = max_depth
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.batch_norm = batch_norm
        self.layer_activation_fn = layer_activation_fn
        self.mask_activation_fn = mask_activation_fn
        self.dropout_p = dropout_p
        self.use_skip = use_skip
        self.normalize_input = normalize_input
        self.window_type = window_type
        super(TFMaskUNet, self).__init__()

        if self.normalize_input:
            self.input_norm = nn.BatchNorm2d(self.num_bins)

        self.autoencoder = AutoEncoder2d(
            num_targets=1,
            num_bins=self.num_bins,
            num_samples=self.num_samples,
            num_channels=self.num_channels,
            max_depth=max_depth,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            block_size=block_size,
            downsampler=downsampler,
            upsampler=upsampler,
            batch_norm=batch_norm,
            activation=layer_activation_fn,
            dropout_p=dropout_p,
            use_skip=use_skip,
        )

        self.mask_activation = _get_activation(
            activation_fn=mask_activation_fn
        )

        if window_type is not None:
            window_fn = _make_hann_window(
                window_length=window_size,
                trainable=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            window_fn = None

        self.data_transform = {
            "n_fft": self.num_fft,
            "hop_length": hop_length,
            "win_length": window_size,
            "window": window_fn,
            "onesided": True,
            "return_complex": True,
        }

        self.stft = lambda data: torch.stft(
            input=data, **self.data_transform
        )
        self.istft = lambda data: torch.istft(
            input=data, **self.data_transform
        )

    def fast_fourier(
        self, mixture_data: torch.Tensor, target_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method to transform raw data to complex-valued STFT data."""
        mixture_stft, target_stft = [], []
        n_frames = mixture_data.size(-1)

        for i in range(self.num_channels):
            mixture_stft.append(
                self.stft(mixture_data[:, i, :].view(-1, n_frames))
            )
            sources_stack = []
            for j in range(self.num_targets):
                sources_stack.append(
                    self.stft(target_data[:, i, :, j].view(-1, n_frames))
                )
            target_stft.append(torch.stack(sources_stack, dim=-1))

        mixture_stft = torch.stack(mixture_stft, dim=1)
        target_stft = torch.stack(target_stft, dim=1)
        return mixture_stft, target_stft

    def process_input(
        self, mixture_data: torch.Tensor, target_data: torch.Tensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Processes audio data into magnitudes spectrograms for training.

        Args:
            mixture_data (Tensor): Mixture input data.
            target_data (Tensor): Target sources data.

        Returns:
            (Tensor, Tensor): Magnitude spectrograms of input mixture and
                source targets.
        """
        mixture_stft, target_stft = self.fast_fourier(
            mixture_data=mixture_data, target_data=target_data
        )
        mixture_spec = torch.abs(mixture_stft).float()
        target_spec = torch.abs(target_stft).float()
        return mixture_spec, target_spec

    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method."""
        if self.normalize_input:
            data = self.input_norm(data.permute(0, 2, 3, 1))
            data = data.permute(0, 3, 1, 2)
        data = self.autoencoder(data)
        mask = self.mask_activation(data)
        return mask

    def backward(self, mask: torch.FloatTensor, target_data: torch.FloatTensor):
        pass

    def update_params(self):
        pass

    def separate(self, audio):
        pass

    def inference(self, audio):
        pass

    def validate(self):
        pass


# class UNetRecurrentMaskTF(UNetTFMaskEstimate):
#     """U-Net source separation model in the time-frequency domain.
#
#     Estimates the target sources directly, rather than using the
#     `soft-masking` technique implemented by the UNetTFMaskEstimate class.
#     Useful for separating multiple sources at once, rather than using a
#     different model to separate each target source.
#
#
#     """
#     __doc__ += UNetTFMaskEstimate.__doc__
#
#     def __init__(self, num_targets: int, **kwargs):
#         self.num_targets = num_targets
#         super(UNetTFSourceEstimate, self).__init__(**kwargs)
#
#
#
# class UNetVAESpec(TFMaskModelBase):
#     """U-Net source separation model in the time-frequency domain."""
#
#     def __init__(
#             self,
#             num_targets: int,
#             num_bins: int,
#             num_samples: int,
#             num_channels: int,
#             num_fft: int,
#             hop_length: int,
#             window_size: int,
#             max_depth: int,
#             hidden_size: int,
#             latent_size: int,
#             kernel_size: Union[Tuple, int],
#             block_size: int = 3,
#             downsampler: str = "max_pool",
#             upsampler: str = "transpose",
#             batch_norm: bool = True,
#             layer_activation: str = "relu",
#             dropout_p: float = 0,
#             use_skip: bool = True,
#             normalize_input: bool = False,
#             mask_activation: str = "relu",
#     ):
#         super(UNetVAESpec, self).__init__(
#             num_bins=num_bins,
#             num_samples=num_samples,
#             num_channels=num_channels,
#             num_fft=num_fft,
#             hop_length=hop_length,
#             window_size=window_size,
#             num_targets=num_targets,
#         )
#
#         self.max_depth = max_depth
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size
#         self.kernel_size = kernel_size
#         self.block_size = block_size
#         self.downsampler = downsampler
#         self.upsampler = upsampler
#         self.batch_norm = batch_norm
#         self.layer_activation = layer_activation
#         self.dropout_p = dropout_p
#         self.use_skip = use_skip
#         self.normalize_input = normalize_input
#         self.mask_activation = mask_activation
#
#         if self.normalize_input:
#             self.input_norm = nn.BatchNorm2d(num_bins)
#
#         self.autoencoder = VAE2d(
#             latent_size=latent_size,
#             num_targets=num_targets,
#             num_bins=num_bins,
#             num_samples=num_samples,
#             num_channels=num_channels,
#             max_depth=max_depth,
#             hidden_size=hidden_size,
#             kernel_size=kernel_size,
#             block_size=block_size,
#             downsampler=downsampler,
#             upsampler=upsampler,
#             batch_norm=batch_norm,
#             activation=layer_activation,
#             dropout_p=dropout_p,
#             use_skip=use_skip,
#         )
#         self.mask_activation = _get_activation(activation_fn=mask_activation)
#
#     def forward(
#             self, data: torch.FloatTensor
#     ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
#         """Forward method."""
#         data = self.input_norm(data) if self.normalize_input else data
#         data = data.permute(0, 3, 1, 2)
#         output, latent_dist = self.autoencoder(data)
#         mask = self.mask_activation(output)
#         mask = mask.permute(0, 2, 3, 1, 4)
#         return mask, latent_dist
