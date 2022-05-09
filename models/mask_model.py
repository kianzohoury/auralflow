import importlib
from os import truncate
from typing import Callable
from collections import OrderedDict

import torch
from torch import Tensor, FloatTensor
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_num_frames, get_stft
from visualizer import log_spectrograms, log_audio
from .base import SeparationModel


class SpectrogramMaskModel(SeparationModel):
    """Spectrogram-domain deep mask estimation model."""

    mixtures: Tensor
    targets: Tensor
    estimates: Tensor
    mask: FloatTensor
    stft: Callable
    inv_stft: Callable

    def __init__(self, configuration: dict):
        super(SpectrogramMaskModel, self).__init__(configuration)
        dataset_params = configuration["dataset_params"]
        self.layer_params = configuration["model_params"]["layer_params"]

        # Calculate number of frames (temporal dimension).
        self.num_samples = get_num_frames(
            sample_rate=dataset_params["sample_rate"],
            sample_length=dataset_params["sample_length"],
            num_fft=dataset_params["num_fft"],
            window_size=dataset_params["window_size"],
            hop_length=dataset_params["hop_length"],
        )

        # Note that number of bins is num_fft // 2 + 1 due to symmetry.
        self.n_fft_bins = configuration["dataset_params"]["num_fft"] // 2 + 1
        self.num_channels = configuration["dataset_params"]["num_channels"]

        # Retrieve class of underlying model architecture.
        arch_constructor = getattr(
            importlib.import_module("models.architectures", "models"),
            configuration["model_params"]["model_type"],
        )

        # Create an instance of the model set to the current device.
        self.model = arch_constructor(
            num_fft_bins=self.n_fft_bins,
            num_samples=self.num_samples,
            num_channels=self.num_channels,
            hidden_dim=self.layer_params["hidden_size"],
            mask_act_fn=configuration["model_params"]["mask_activation"],
            leak_factor=self.layer_params["leak_factor"],
            normalize_input=configuration["model_params"]["normalize_input"],
        ).to(self.device)

        # Define the specified short-time fourier transform and its inverse.
        self.stft = get_stft(
            num_fft=dataset_params["num_fft"],
            hop_length=dataset_params["hop_length"],
            window_size=dataset_params["window_size"],
            use_hann=dataset_params["use_hann_window"],
            trainable=dataset_params["learn_filterbanks"],
            inverse=False,
            device=self.device,
        )
        self.inv_stft = get_stft(
            num_fft=dataset_params["num_fft"],
            hop_length=dataset_params["hop_length"],
            window_size=dataset_params["window_size"],
            use_hann=dataset_params["use_hann_window"],
            trainable=dataset_params["learn_filterbanks"],
            inverse=True,
            device=self.device,
        )

        # Define loss, optimizer, etc.
        if self.training_mode:
            lr = self.config["training_params"]["lr"]
            self.optimizer = AdamW(self.model.parameters(), lr)
            self.train_losses = []
            self.val_losses = []
            self.criterion = self.model.loss_fn
            self.patience = self.config["training_params"]["stop_patience"]
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", verbose=True
            )
            # self.accum_steps = 10

    @staticmethod
    def fast_fourier(transform: Callable, audio: Tensor) -> Tensor:
        """Transforms raw audio to complex-valued STFT audio."""
        audio_stft = []
        n_batch, n_channels, n_frames, n_targets = audio.size()

        for i in range(n_channels):
            sources_stack = []
            for j in range(n_targets):
                sources_stack.append(
                    transform(audio[:, i, :, j].reshape((n_batch, n_frames)))
                )
            audio_stft.append(torch.stack(sources_stack, dim=-1))

        data_stft = torch.stack(audio_stft, dim=1)
        return data_stft

    @staticmethod
    def inverse_fast_fourier(transform: Callable, complex_stft: Tensor):
        """Transforms complex-valued STFT audio to temporal audio domain."""
        source_estimate = []
        n_batch, n_channels, n_frames, n_targets = complex_stft.size()

        for i in range(n_targets):
            source_estimate.append(
                transform(complex_stft[:, :, :, i].squeeze(-1))
            )

        source_estimate = torch.stack(source_estimate, dim=-1)
        return source_estimate

    def process_audio(self, audio: Tensor, magnitude: bool = True) -> Tensor:
        """Performs FFT algorithm and returns mag or complex spectrograms."""
        data_stft = self.fast_fourier(
            transform=self.stft, audio=audio.to(self.device)
        )
        return torch.abs(data_stft) if magnitude else data_stft

    def set_data(self, mixture: Tensor, target: Tensor) -> None:
        """Wrapper method processes and sets data for internal access."""
        self.mixtures = self.process_audio(mixture).squeeze(-1)
        self.targets = self.process_audio(target).squeeze(-1)

    def forward(self):
        """Performs target source estimation by applying a learned soft-mask.

        Target S is acquired by taking the Hadamard product between the mixture
        signal X,and the output of the network, M (estimated soft-mask), such
        that S = M * X.
        """
        self.mask = self.model(self.mixtures)
        self.estimates = self.mask * self.mixtures

    def backward(self):
        """Computes batch-wise loss between estimate and target sources."""
        self.batch_loss = self.criterion(self.estimates, self.targets)

    def optimizer_step(self):
        """Performs gradient computation and parameter optimization."""
        self.batch_loss.backward()
        self.optimizer.step()
        for param in self.model.parameters():
            param.grad = None

    def scheduler_step(self):
        """Decreases learning rate if validation loss does not improve."""
        self.scheduler.step(self.val_losses[-1])

    def stop_early(self):
        """Signals that training should stop based on patience criteria."""
        if len(self.val_losses) <= 1:
            return False
        elif self.val_losses[-1] >= min(self.val_losses[:-1]):
            self.patience -= 1
            if not self.patience:
                return True
        else:
            self.patience = self.config["training_params"]["stop_patience"]
        return False

    def separate(self, audio: Tensor) -> Tensor:
        """Applies inv STFT to target source to retrieve time-domain signal.

        * Takes the target source S = M * X resultant from the forward pass,
        applies phase correction, and applies the inverse fourier transform to
        yield the separated audio with the shape (n_channels, n_samples). The
        detailed procedure is as follows:

        X = |STFT(A)|
        S = X * M
        S_p = X * M * P
        A_s = iSTFT(S_p)

        * where X: magnitude spectrogram;
        * S: output of forward pass
        * S_p: phase corrected output S
        * P: complex-valued phase matrix, i.e., exp^(i * theta), where theta
          is the angle between the real and imaginary parts of STFT(A).
        * A_s: estimate source signal converted from time-freq to time-only
          domain.
        """
        complex_stft = self.process_audio(audio, magnitude=False).squeeze(-1)
        mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
        self.mixtures = mag
        self.test()
        phase_corrected = self.estimates * torch.exp(1j * phase)
        source_estimate = self.inverse_fast_fourier(
            self.inv_stft, phase_corrected.permute(0, 2, 3, 1)
        )
        return source_estimate

    def post_epoch_callback(
        self,
        mixture_audio: Tensor,
        target_audio: Tensor,
        writer: SummaryWriter,
        global_step: int,
    ):
        """Logs spectrogram images and separated audio after each epoch."""
        target_labels = sorted(self.config["dataset_params"]["targets"])
        target_name = target_labels[0]
        log_spectrograms(
            writer=writer,
            global_step=global_step,
            audio_data=OrderedDict(
                [
                    ("mixture", self.mixtures.unsqueeze(-1)),
                    (f"{target_name}_estimate", self.estimates.unsqueeze(-1)),
                    (f"{target_labels[0]}_true", self.targets.unsqueeze(-1)),
                ]
            ),
            sample_rate=self.config["dataset_params"]["sample_rate"],
        )
        log_audio(
            writer=writer,
            global_step=global_step,
            estimate_data=self.separate(mixture_audio).unsqueeze(-1),
            target_data=target_audio.permute(0, 2, 1, 3),
            target_labels=sorted(self.config["dataset_params"]["targets"]),
            sample_rate=self.config["dataset_params"]["sample_rate"],
        )

    def get_batch_loss(self) -> Tensor:
        return self.batch_loss.item()
