import importlib
from typing import Callable
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torch.optim import AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_num_frames, get_stft, AudioTransform
from visualizer import log_spectrograms, log_audio, log_gradients
from .base import SeparationModel


class SpectrogramMaskModel(SeparationModel):
    """Spectrogram-domain deep mask estimation model."""

    mixtures: Tensor
    targets: Tensor
    estimates: Tensor
    residuals: Tensor
    mask: FloatTensor

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

        # Note that num bins will be num_fft // 2 + 1 due to symmetry.
        self.n_fft_bins = configuration["dataset_params"]["num_fft"] // 2 + 1
        self.num_channels = configuration["dataset_params"]["num_channels"]

        # Retrieve class name of the requested model architecture.
        model_name = getattr(
            importlib.import_module("models.architectures", "models"),
            configuration["model_params"]["model_type"],
        )

        # Create the model instance and set to current device.
        self.model = model_name(
            num_fft_bins=self.n_fft_bins,
            num_samples=self.num_samples,
            num_channels=self.num_channels,
            hidden_dim=self.layer_params["hidden_size"],
            mask_act_fn=configuration["model_params"]["mask_activation"],
            leak_factor=self.layer_params["leak_factor"],
            normalize_input=configuration["model_params"]["normalize_input"],
            residual=configuration["model_params"]["learn_residual"]
        ).to(self.device)

        # Instantiate data transformer.
        self.transform = AudioTransform(
            num_fft=dataset_params["num_fft"],
            hop_length=dataset_params["hop_length"],
            window_size=dataset_params["window_size"],
            power=2 if dataset_params["power_spectrum"] else 1
        )

        # self.stft = get_stft(
        #     num_fft=dataset_params["num_fft"],
        #     hop_length=dataset_params["hop_length"],
        #     window_size=dataset_params["window_size"],
        #     use_hann=dataset_params["use_hann_window"],
        #     trainable=dataset_params["learn_filterbanks"],
        #     inverse=False,
        #     device=self.device,
        # )
        # self.inv_stft = get_stft(
        #     num_fft=dataset_params["num_fft"],
        #     hop_length=dataset_params["hop_length"],
        #     window_size=dataset_params["window_size"],
        #     use_hann=dataset_params["use_hann_window"],
        #     trainable=dataset_params["learn_filterbanks"],
        #     inverse=True,
        #     device=self.device,
        # )

        self.model.set_criterion(nn.L1Loss())

        # Define loss, optimizer, etc.
        if self.training_mode:
            lr = self.config["training_params"]["lr"]
            self.optimizer = AdamW(self.model.parameters(), lr)
            self.train_losses = []
            self.val_losses = []
            self.criterion = self.model.loss_fn
            self.patience = self.config["training_params"]["stop_patience"]
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", verbose=True, patience=6
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

    def set_data(self, mixture: Tensor, target: Tensor) -> None:
        """Wrapper method processes and sets data for internal access."""
        self.mixtures = torch.abs(
            self.transform.to_spectrogram(mixture.to(self.device))
        ).squeeze(-1)
        self.targets = torch.abs(
            self.transform.to_spectrogram(target.to(self.device))
        )

    def forward(self):
        """Estimates target source by applying the learned mask to the mixture.

        * Target source S' is acquired by taking the Hadamard product between
          the mixture signal X, and the output of the network, M
          (estimated soft-mask), such that S = M * X.

        * If learn_residual is True, network will also estimate the residual
          signal separately.
        """
        self.mask = self.model(self.mixtures)
        self.estimates = self.mask * self.mixtures
        self.residuals = self.model.residual_mask * self.mixtures

    def get_loss(self) -> float:
        """Computes batch-wise loss."""
        self.batch_loss = self.criterion(self.estimates, self.targets) + self.criterion(
            self.residuals, self.mixtures - self.targets
        )
        return self.batch_loss.item()

    def backward(self) -> None:
        """Performs gradient computation and backpropagation."""
        self.batch_loss.backward()

    def optimizer_step(self) -> None:
        """Updates model's parameters."""
        self.optimizer.step()
        self.optimizer.zero_grad()
        # for param in self.model.parameters():
        #     param.grad = None

    def scheduler_step(self) -> None:
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
        """Applies inv STFT to target source to retrieve time-domain signal."""
        # Compute complex-valued stft.
        complex_stft = self.transform.to_spectrogram(
            audio.to(self.device)
        ).squeeze(-1)
        # Extract magnitude and phase separately.
        mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
        # Get source estimate s' = mask * mixture.
        self.mixtures = mag.to(self.device)
        self.test()
        # Apply phase correction and transform back to time domain.
        phase_corrected = self.estimates * torch.exp(1j * phase)
        source_estimate = self.transform.to_audio(
            phase_corrected.permute(0, 2, 3, 1)
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

        self.eval()
        estimate_audio = self.separate(mixture_audio.clone())
        estimate_mel_spec = self.transform.to_mel_scale(
            self.estimates, to_db=True
        )
        target_mel_spec = self.transform.audio_to_mel(target_audio)

        log_spectrograms(
            writer=writer,
            global_step=global_step,
            estimate_spec=estimate_mel_spec,
            target_spec=target_mel_spec,
            estimate_audio=estimate_audio,
            target_audio=target_audio,
            target_labels=target_labels,
            sample_rate=self.config["dataset_params"]["sample_rate"]
        )

        log_audio(
            writer=writer,
            global_step=global_step,
            estimate_data=estimate_audio.unsqueeze(-1),
            target_data=target_audio.permute(0, 2, 1, 3),
            target_labels=sorted(self.config["dataset_params"]["targets"]),
            sample_rate=self.config["dataset_params"]["sample_rate"],
        )
