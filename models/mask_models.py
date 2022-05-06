import inspect

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from validate import cross_validate
from validate import cross_validate

from typing import Union, Tuple, Any, List, Callable
from .modules import AutoEncoder2d
from .layers import _get_activation
from .base import SeparationModel
import torch.nn as nn
from .static_models import (
    SpectrogramNetSimple,
    SpectrogramLSTM,
    SpectrogramLSTMVariational,
)
from visualizer import log_residual_specs
from losses import vae_loss
from utils.data_utils import get_num_frames, get_stft, get_inverse_stft
from torch import Tensor, FloatTensor


class SpectrogramMaskModel(SeparationModel):
    """"""

    mixtures: Tensor
    targets: Tensor
    estimates: Tensor
    mask: FloatTensor
    stft: Callable
    inv_stft: Callable

    def __init__(self, configuration: dict):
        dataset_params = configuration["dataset_params"]
        arch_params = configuration["architecture_params"]
        num_samples = get_num_frames(
            sample_rate=dataset_params["sample_rate"],
            sample_length=dataset_params["sample_length"],
            num_fft=dataset_params["num_fft"],
            window_size=dataset_params["window_size"],
            hop_length=dataset_params["hop_length"],
        )
        super(SpectrogramMaskModel, self).__init__(configuration)
        self.model = SpectrogramLSTMVariational(
            num_fft_bins=configuration["dataset_params"]["num_fft"] // 2 + 1,
            num_samples=num_samples,
            num_channels=configuration["dataset_params"]["num_channels"],
            lstm_hidden_size=1024,
        )
        # self.model = SpectrogramLSTM(
        #     num_fft_bins=configuration["dataset_params"]["num_fft"] // 2 + 1,
        #     num_samples=num_samples,
        #     num_channels=configuration["dataset_params"]["num_channels"],
        #     lstm_hidden_size=1024
        # )
        # num_models = len(configuration["dataset_params"]["targets"])

        self.model = self.model.to(self.device)

        self.stft = get_stft(
            num_fft=dataset_params["num_fft"],
            window_size=dataset_params["window_size"],
            hop_length=dataset_params["hop_length"],
        )

        self.inv_stft = get_inverse_stft(
            num_fft=dataset_params["num_fft"],
            window_size=dataset_params["window_size"],
            hop_length=dataset_params["hop_length"],
        )

        if self.training_mode:
            lr = self.config["training_params"]["lr"]
            self.optimizer = AdamW(self.model.parameters(), lr)
            self.train_losses = []
            self.val_losses = []
            self.criterion = nn.L1Loss()
            self.stop_patience = self.config["training_params"][
                "stop_patience"
            ]

    @staticmethod
    def fast_fourier(transform: Callable, data: Tensor) -> Tensor:
        """Helper method to transform raw data to complex-valued STFT data."""
        data_stft = []
        n_batch, n_channels, n_frames, n_targets = data.size()

        for i in range(n_channels):
            sources_stack = []
            for j in range(n_targets):
                sources_stack.append(
                    transform(data[:, i, :, j].reshape((n_batch, n_frames)))
                )
            data_stft.append(torch.stack(sources_stack, dim=-1))

        data_stft = torch.stack(data_stft, dim=1)
        return data_stft

    def process_audio(self, audio: Tensor, magnitude: bool = False) -> Tensor:
        """Processes audio data into magnitudes spectrograms."""
        data_stft = self.fast_fourier(transform=self.stft, data=audio)
        return torch.abs(data_stft) if magnitude else data_stft

    def set_data(self, mixture: Tensor, target: Tensor) -> None:
        """Wrapper method calls process_data and transfers tensors to GPU."""
        self.mixtures = self.process_audio(mixture).squeeze(-1).to(self.device)
        self.targets = self.process_audio(target).squeeze(-1).to(self.device)

    def forward(self):
        """Performs forward pass to estimate the multiplicative soft-mask."""
        self.mask = self.model(self.mixtures)

    def backward(self):
        """Computes batch-wise loss between estimate and target sources."""
        self.estimates = self.mask * self.mixtures
        self.batch_loss = (
            self.criterion(self.estimates, self.targets)
            + self.model.get_kl_div()
        )

    def optimizer_step(self):
        """Performs gradient computation and parameter optimization."""
        self.batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def stop_early(self):
        """Signals that training should stop based on patience criteria."""
        if len(self.val_losses) <= 1:
            return False
        elif self.val_losses[-1] >= self.val_losses[-2]:
            self.stop_patience -= 1
            if not self.stop_patience:
                return True
        else:
            self.stop_patience = self.config["training_params"][
                "stop_patience"
            ]
        return False

    # def validate(self, val_dataloader: DataLoader, max_iters: int):
    #     cross_validate(
    #         model=self.model,
    #         val_dataloader=val_dataloader,
    #         writer=writer,
    #         max_iters=max_iters,
    #         global_step=global_step
    #     )

    def separate(self, audio):
        """Separates audio and converts stft back to time domain."""
        complex_stft = self.process_audio(audio, magnitude=False)
        mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
        mask = self.model(mag)
        estimate = mask * mag
        phase_corrected = estimate * torch.exp(1j * phase)
        return phase_corrected

    def post_epoch_callback(
        self,
        writer: SummaryWriter,
        global_step: int,
        val_dataloader: DataLoader,
        max_iters: int,
    ):
        """Called at the end of each epoch."""
        # val_step = self.config['max_iters'] * len(self.train_losses)

        # cross_validate(self, val_dataloader, max_iters, writer)

        log_residual_specs(
            writer=writer,
            global_step=global_step,
            estimate_data=self.estimates.unsqueeze(-1),
            target_data=self.targets.unsqueeze(-1),
            target_labels=self.config["dataset_params"]["targets"],
            sample_rate=self.config["dataset_params"]["sample_rate"],
        )

    def get_batch_loss(self):
        return self.batch_loss.item()

    # def estimate_audio(self, audio):
    #     with torch.no_grad():
    #         mixture_data = self.stft(audio).unsqueeze(0)
    #         mixture_mag_data = torch.abs(mixture_data).float()
    #         mixture_phase_data = torch.angle(mixture_data)
    #         source_mask = self.forward(mixture_mag_data)
    #         estimate_mag_data = source_mask * mixture_mag_data
    #         estimate_phase_corrected = estimate_mag_data * torch.exp(
    #             1j * mixture_phase_data
    #         )
    #     estimate_signal = self.istft(estimate_phase_corrected)
    #     return estimate_signal


#
# if window_type is not None:
#     window_fn = _make_hann_window(
#         window_length=window_size,
#         trainable=False,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#     )
# else:
#     window_fn = None
#


#         hop_length (int): Hop length.
#         window_size (int): Window size.
#         window_type (optional[str]): Windowing function for the stft transform.
#         Default: 'hann'.
#     trainable_window (bool): Whether to learn the windowing function.
#     Default: False.

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
