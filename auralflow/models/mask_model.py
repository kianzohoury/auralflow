# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import copy
import torch

from .base import SeparationModel
from torch import Tensor, FloatTensor
from typing import Optional
from auralflow.utils.data_utils import get_num_stft_frames, AudioTransform


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
        self.num_stft_frames = get_num_stft_frames(
            sample_len=self.dataset_params["sample_length"],
            sr=self.dataset_params["sample_rate"],
            win_size=self.dataset_params["window_size"],
            hop_len=self.dataset_params["hop_length"],
            center=True,
        )

        # Note that num bins will be num_fft // 2 + 1 due to symmetry.
        self.n_fft_bins = self.dataset_params["num_fft"] // 2 + 1
        self.num_channels = self.dataset_params["num_channels"]
        self.target_labels = sorted(self.dataset_params["targets"])
        self.multi_estimator = len(self.target_labels) > 1

        # Create the model instance and set to current device.
        self.model = self.base_model_type(
            num_fft_bins=self.n_fft_bins,
            num_frames=self.num_stft_frames,
            num_channels=self.num_channels,
            hidden_channels=self.model_params["hidden_channels"],
            mask_act_fn=self.model_params["mask_activation"],
            leak_factor=self.model_params["leak_factor"],
            dropout_p=self.model_params["dropout_p"],
            normalize_input=self.model_params["normalize_input"],
            normalize_output=self.model_params["normalize_output"],
            device=self.device,
        ).to(self.device)

        # Instantiate data transformer for pre/post audio processing.
        self.transform = AudioTransform(
            num_fft=self.dataset_params["num_fft"],
            hop_length=self.dataset_params["hop_length"],
            window_size=self.dataset_params["window_size"],
            device=self.device,
        )

        self.scale = 1
        self.is_best_model = True
        # self.f32_weights = self.copy_params(self.model)

    @staticmethod
    def copy_params(src_module):
        params_dest = {}
        for name, param in src_module.named_parameters():
            params_dest[name] = copy.deepcopy(param.data)
            param = param.to(dtype=torch.float16)
        return params_dest

    def update_f32_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.f32_weights[name].grad = param.grad.to(
                    dtype=torch.float32
                )
                self.f32_weights[name].grad /= self.scale
        self.model._parameters = self.f32_weights

    def set_data(self, mix: Tensor, target: Optional[Tensor] = None) -> None:
        """Wrapper method processes and sets data for internal access."""
        # Compute complex-valued STFTs and send tensors to GPU if available.
        mix_complex_stft = self.transform.to_spectrogram(mix.to(self.device))
        if target is not None:
            target_complex_stft = self.transform.to_spectrogram(
                target.squeeze(-1).to(self.device)
            )
            self.target = torch.abs(target_complex_stft)

        # Separate magnitude and phase.
        self.mixture = torch.abs(mix_complex_stft)
        self.phase = torch.angle(mix_complex_stft)

    def forward(self) -> None:
        """Estimates target by applying the learned mask to the mixture."""
        self.mask = self.model(self.mixture)
        self.estimate = self.mask * (self.mixture.clone().detach())

    def separate(self, audio: Tensor) -> Tensor:
        """Transforms and returns source estimate in the audio domain."""
        # Set data and estimate source.
        self.set_data(mix=audio)
        self.test()

        # Apply phase correction to estimate.
        phase_corrected = self.estimate * torch.exp(1j * self.phase)
        target_estimate = self.transform.to_audio(phase_corrected)
        return target_estimate

    def compute_loss(self) -> float:
        """Updates and returns the current batch-wise loss."""
        self.criterion()
        # Apply scaling.
        self.batch_loss = self.scale * self.batch_loss
        return self.batch_loss.item()

    def backward(self) -> None:
        """Performs gradient computation and backpropagation."""
        self.batch_loss.backward()

    def optimizer_step(self) -> None:
        """Updates model's parameters."""
        self.train()
        # skip_update = False
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         if param.grad.isnan().any() or param.grad.isinf().any():
        #             skip_update = True
        #             print(name)
        # if not skip_update:
        #     self.update_f32_gradients()
            # self.optimizer.step()
        self.optimizer.step()

        # grad_norm = nn.utils.clip_grad_norm_(
        #     self.model.parameters(), max_norm=100
        # )

        # self.grad_scaler.unscale_(self.optimizer)
        # grad_norm = nn.utils.clip_grad_norm_(self.f32_weights, max_norm=2e10)
        # print(grad_norm)

        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         weight_norm = torch.linalg.norm(param)
        #         grad_norm = torch.linalg.norm(param.grad)
        #         print(f"weight norm: {weight_norm} \n grad norm {grad_norm}")

        # Quicker gradient zeroing.
        for param in self.model.parameters():
            param.grad = None

        # self.f32_weights = self.copy_params(self.model)

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
            self.max_lr_steps -= 1 if not self.stop_patience else 0
            self.is_best_model = False
        return not self.max_lr_steps

