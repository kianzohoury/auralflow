import importlib

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torch.optim import AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_num_frames, AudioTransform
from visualizer import visualize_audio, listen_audio
from .base import SeparationModel
from losses import residual_loss


class SpectrogramMaskModel(SeparationModel):
    """Spectrogram-domain deep mask estimation model."""

    mixtures: Tensor
    phase: Tensor
    targets: Tensor
    estimates: FloatTensor
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
            residual=configuration["model_params"]["learn_residual"],
        ).to(self.device)

        # Instantiate data transformer.
        self.transform = AudioTransform(
            num_fft=dataset_params["num_fft"],
            hop_length=dataset_params["hop_length"],
            window_size=dataset_params["window_size"],
            device=self.device,
        )

        self.input_norm = configuration["model_params"]["normalize_input"]

        self.target_labels = sorted(self.config["dataset_params"]["targets"])
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

    def set_data(self, mixture: Tensor, target: Tensor) -> None:
        """Wrapper method processes and sets data for internal access."""
        # Compute complex-valued STFTs and send tensors to GPU if available.
        mix_complex_stft = self.transform.to_spectrogram(
            mixture.squeeze(-1).to(self.device)
        )
        target_complex_stft = self.transform.to_spectrogram(
            target.permute(0, 3, 1, 2).to(self.device)
        ).permute(0, 2, 3, 4, 1)

        # Separate magnitude and phase, and store data for internal access.
        self.mixtures = torch.abs(mix_complex_stft)
        self.targets = torch.abs(target_complex_stft)
        self.phase = torch.angle(mix_complex_stft)

    def forward(self) -> None:
        """Estimates target source by applying the learned mask to the mixture.

        Target source S' is acquired by taking the Hadamard product between
        the mixture signal X, and the output of the network, M
        (estimated soft-mask), such that S = M * X. If learning residual,
        network will estimate it separately.
        """
        # if self.input_norm:
        #     input_data = self.mixtures / torch.max(self.mixtures)
        # else:
        #     input_data = self.mixtures
        # print(self.mixtures.shape)
        self.mask = self.model(self.mixtures)
        self.estimates = self.mask * self.mixtures
        # self.residuals = self.model.residual_mask * self.mixtures.clone().detach()

    def get_loss(self) -> float:
        """Computes batch-wise loss."""
        # print(self.targets.shape, self.residuals.shape, self.mixtures.shape, self.estimates.shape)
        self.batch_loss = self.criterion(self.estimates,
            self.targets.squeeze(-1))
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
        # Compute complex-valued STFTs and send tensors to GPU if avaialable.
        mix_complex_stft = self.transform.to_spectrogram(
            audio.squeeze(-1).to(self.device)
        )

        # Separate magnitude and phase.
        self.mixtures = torch.abs(mix_complex_stft)
        self.phase = torch.angle(mix_complex_stft)

        # Get source estimate s' = mask * mixture and apply phase correction.
        self.test()
        phase_corrected = self.estimates * torch.exp(1j * self.phase)
        target_estimate = self.transform.to_audio(phase_corrected)
        return target_estimate

    def post_epoch_callback(
        self,
        mixture_audio: Tensor,
        target_audio: Tensor,
        writer: SummaryWriter,
        global_step: int,
    ):
        """Logs spectrogram images and separated audio after each epoch."""
        # Visualize and listen to audio via tensorboard.
        visualize_audio(
            model=self,
            mixture_audio=mixture_audio[0].unsqueeze(0),
            target_audio=target_audio[0],
            to_tensorboard=True,
            writer=writer,
            save_images=False,
            global_step=global_step,
        )
        listen_audio(
            model=self,
            mixture_audio=mixture_audio[0].unsqueeze(0),
            target_audio=target_audio[0],
            writer=writer,
            global_step=global_step,
            residual=True,
            sample_rate=self.config["dataset_params"]["sample_rate"],
        )
