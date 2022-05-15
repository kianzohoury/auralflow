import importlib

import torch
from torch import Tensor, FloatTensor
from torch.optim import AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from losses import get_model_criterion
from utils.data_utils import get_num_frames, AudioTransform
from visualizer import visualize_audio, listen_audio, log_gradients
from .base import SeparationModel


class SpectrogramMaskModel(SeparationModel):
    """Spectrogram-domain deep mask estimation model."""

    mixture: Tensor
    phase: Tensor
    target: Tensor
    estimate: FloatTensor
    residual: Tensor
    mask: FloatTensor

    def __init__(self, configuration: dict):
        super(SpectrogramMaskModel, self).__init__(configuration)

        # Calculate number of frames (temporal dimension).
        self.num_samples = get_num_frames(
            sample_rate=self.dataset_params["sample_rate"],
            sample_length=self.dataset_params["sample_length"],
            num_fft=self.dataset_params["num_fft"],
            window_size=self.dataset_params["window_size"],
            hop_length=self.dataset_params["hop_length"],
        )

        # Note that num bins will be num_fft // 2 + 1 due to symmetry.
        self.n_fft_bins = self.dataset_params["num_fft"] // 2 + 1
        self.num_channels = self.dataset_params["num_channels"]
        self.target_labels = sorted(self.dataset_params["targets"])
        self.multi_estimator = len(self.target_labels) > 1

        # Retrieve class name of the requested model architecture.
        model_name = getattr(
            importlib.import_module("models.architectures", "models"),
            self.model_params["model_type"],
        )

        # Create the model instance and set to current device.
        self.model = model_name(
            num_fft_bins=self.n_fft_bins,
            num_samples=self.num_samples,
            num_channels=self.num_channels,
            hidden_dim=self.model_params["hidden_size"],
            mask_act_fn=self.model_params["mask_activation"],
            leak_factor=self.model_params["leak_factor"],
            normalize_input=self.model_params["normalize_input"],
        ).to(self.device)

        # Instantiate data transformer for pre/post audio processing.
        self.transform = AudioTransform(
            num_fft=self.dataset_params["num_fft"],
            hop_length=self.dataset_params["hop_length"],
            window_size=self.dataset_params["window_size"],
            device=self.device,
        )

        if self.training_mode:
            # Set model criterion.
            self.criterion = get_model_criterion(
                model=self, config=configuration
            )
            # Load optimizer.
            self.optimizer = AdamW(
                self.model.parameters(), self.training_params["lr"]
            )
            # Load lr scheduler.
            self.stop_patience = self.training_params["stop_patience"]
            self.max_lr_reductions = self.training_params["max_lr_reductions"]
            self.is_best_model = True
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                verbose=True,
                patience=self.stop_patience,
            )
            # Store train/val losses.
            self.train_losses = []
            self.val_losses = []

    def set_data(self, mixture: Tensor, target: Tensor) -> None:
        """Wrapper method processes and sets data for internal access."""
        # Drop last dimension if only estimating one target source.
        mixture = mixture.squeeze(-1) if not self.multi_estimator else mixture
        target = target.squeeze(-1) if not self.multi_estimator else target

        # Compute complex-valued STFTs and send tensors to GPU if available.
        mix_complex_stft = self.transform.to_spectrogram(
            mixture.to(self.device)
        )
        target_complex_stft = self.transform.to_spectrogram(
            target.to(self.device)
        )

        # Separate magnitude and phase, and store data for internal access.
        self.mixture = torch.abs(mix_complex_stft)
        self.target = torch.abs(target_complex_stft)
        self.phase = torch.angle(mix_complex_stft)

    def forward(self) -> None:
        """Estimates target by applying the learned mask to the mixture."""
        self.mask = self.model(self.mixture)
        self.estimate = self.mask * (self.mixture.clone().detach())

    def compute_loss(self) -> float:
        """Updates and returns the current batch-wise loss."""
        self.criterion()
        return self.batch_loss.item()

    def backward(self) -> None:
        """Performs gradient computation and backpropagation."""
        self.batch_loss.backward()

    def optimizer_step(self) -> None:
        """Updates model's parameters."""
        self.train()
        self.optimizer.step()
        # Quicker gradient zeroing.
        for param in self.model.parameters():
            param.grad = None

    def scheduler_step(self) -> bool:
        """Reduces lr if val loss does not improve, and signals early stop."""
        self.scheduler.step(self.val_losses[-1])
        prev_loss = min(self.val_losses[:-1], default=float("inf"))
        delta = prev_loss - self.val_losses[-1]

        if delta > 0:
            self.stop_patience = self.training_params["stop_patience"]
            self.is_best_model = True
        else:
            self.stop_patience -= 1
            self.max_lr_reductions -= 1 if not self.stop_patience else 0
            self.is_best_model = False
        return not self.max_lr_reductions

    def separate(self, audio: Tensor) -> Tensor:
        """Applies inv STFT to target source to retrieve time-domain signal."""
        # Compute complex-valued STFTs and send tensors to GPU if available.
        mix_complex_stft = self.transform.to_spectrogram(audio.to(self.device))

        # Separate magnitude and phase.
        self.mixture = torch.abs(mix_complex_stft)
        self.phase = torch.angle(mix_complex_stft)

        # Get source estimate s' = mask * mixture and apply phase correction.
        self.test()
        phase_corrected = self.estimate * torch.exp(1j * self.phase)
        target_estimate = self.transform.to_audio(phase_corrected)
        return target_estimate

    def mid_epoch_callback(
        self, writer: SummaryWriter, global_step: int
    ) -> None:
        """Called during epoch before parameter update."""
        log_gradients(self.model, writer=writer, global_step=global_step)

    def post_epoch_callback(
        self,
        mixture_audio: Tensor,
        target_audio: Tensor,
        writer: SummaryWriter,
        global_step: int,
    ) -> None:
        """Logs spectrogram images and separated audio after each epoch."""
        save_images = self.visualizer_params["save_images"]
        save_images = save_images and global_step % self.visualizer_params["save_frequency"] == 0
        for i, label in enumerate(self.target_labels):
            visualize_audio(
                model=self,
                label=label,
                mixture_audio=mixture_audio[:, :, :, i],
                target_audio=target_audio[:, :, :, i],
                to_tensorboard=True,
                writer=writer,
                save_images=save_images,
                n_samples=self.visualizer_params["num_images_view"],
                global_step=global_step,
            )
            listen_audio(
                model=self,
                label=label,
                mixture_audio=mixture_audio[:, :, :, i],
                target_audio=target_audio[:, :, :, i],
                writer=writer,
                sample_rate=self.dataset_params["sample_rate"],
                n_samples=self.visualizer_params["num_audio_listen"],
                global_step=global_step,
            )
