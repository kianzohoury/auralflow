# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import importlib
import torch
import torch.nn as nn


from auralflow.transforms import AudioTransform, get_num_stft_frames
from .base import SeparationModel
from torch import FloatTensor, Tensor
from typing import List, Optional


class SpectrogramMaskModel(SeparationModel):
    """Spectrogram-domain deep mask estimation model.

    Args:
        base_model_type (str): Base model architecture.
        target_labels (List[str]): Target source labels.
        num_fft (int): Number of FFT bins. Note that only ``num_fft`` // 2 + 1
            bins will be used due to symmetry. Default: ``1024``.
        window_size (int): Window size. Default: ``1024``.
        hop_length (int): Hop length. Default: ``512``.
        sample_length (int): Duration of audio samples in seconds.
            Default: ``3``.
        sample_rate (int): Sample rate. Default: ``44100``.
        num_channels (int): Number of input/output channels. Default: ``1``.
        num_hidden_channels (int): Number of initial hidden channels.
            Default: ``16``.
        mask_act_fn (str): Mask activation function. Default: ``'sigmoid'``.
        leak_factor (float): Leak factor if ``mask_act_fn='leaky_relu'``.
            Default: ``0``.
        dropout_p (float): Layer dropout probability. Default: ``0.4``.
        normalize_input (bool): Whether to learn input scaling/centering
            parameters. Default: ``True``.
        normalize_output (bool): Whether to learn output scaling/centering
            parameters. Default: ``True``.
        device (str): Device. Default: ``'cpu'``.
    """

    estimate_audio: FloatTensor
    estimate_spec: FloatTensor
    mix_spec: FloatTensor
    mix_phase: FloatTensor
    target_audio: FloatTensor
    target_spec: FloatTensor
    mask: FloatTensor

    def __init__(
        self,
        base_model_type: str,
        target_labels: List[str],
        num_fft: int = 1024,
        window_size: int = 1024,
        hop_length: int = 512,
        sample_length: int = 3,
        sample_rate: int = 44100,
        num_channels: int = 1,
        num_hidden_channels: int = 16,
        mask_act_fn: str = "sigmoid",
        leak_factor: float = 0,
        dropout_p: float = 0.4,
        normalize_input: bool = True,
        normalize_output: bool = True,
        device: str = 'cpu'
    ):
        super(SpectrogramMaskModel, self).__init__()

        self.target_labels = sorted(target_labels)
        self.device = device

        # Calculate number of frames (temporal dimension).
        self.num_stft_frames = get_num_stft_frames(
            sample_len=sample_length,
            sr=sample_rate,
            win_size=window_size,
            hop_len=hop_length,
            center=True,
        )

        # Note that the num bins will be num_fft // 2 + 1 due to symmetry.
        self.n_fft_bins = num_fft // 2 + 1
        self.num_out_channels = num_channels
        self._multi_estimator = len(self.target_labels) > 1
        self._is_best_model = False

        # Retrieve requested base model architecture class.
        base_class = getattr(
            importlib.import_module("auralflow.models"),
            base_model_type
        )

        # Create the model instance and set to current device.
        self.model = base_class(
            num_fft_bins=self.n_fft_bins,
            num_frames=self.num_stft_frames,
            num_channels=self.num_out_channels,
            hidden_channels=num_hidden_channels,
            mask_act_fn=mask_act_fn,
            leak_factor=leak_factor,
            dropout_p=dropout_p,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            device=self.device
        ).to(self.device)

        # Instantiate data transformer for pre/post audio processing.
        self.transform = AudioTransform(
            num_fft=num_fft,
            hop_length=hop_length,
            window_size=window_size,
            device=self.device
        )

    def set_data(self, mix: Tensor, target: Optional[Tensor] = None) -> None:
        """Wrapper method processes and sets data for internal access.

        Transforms mixture audio (and optionally target audio) into
        spectrogram data.

        Args:
            mix (Tensor): Mixture audio data.
            target (Optional[Tensor]): Target audio data.
        """
        # Compute complex-valued STFTs and send tensors to GPU if available.
        mix_complex_stft = self.transform.to_spectrogram(mix.to(self.device))
        if target is not None:
            # target_complex_stft = self.transform.to_spectrogram(
            #     target.squeeze(-1).to(self.device)
            # )
            # # Separate target magnitude and phase.
            # self.target = torch.abs(target_complex_stft)
            self.target_spec = target.squeeze(-1).to(self.device).float()

        # Separate mixture magnitude and phase.
        self.mix_spec = torch.abs(mix_complex_stft)
        self.mix_phase = torch.angle(mix_complex_stft).float()

    def forward(self) -> None:
        """Estimates target by applying the learned mask to the mixture."""
        self.mask = self.model(self.mix_spec)
        self.estimate_spec = self.mask * (self.mix_spec.clone().detach())

        phase_corrected = self.estimate_spec * torch.exp(1j * self.mix_phase)
        self.estimate_audio = self.transform.to_audio(phase_corrected).float()

    def separate(self, audio: Tensor) -> Tensor:
        """Transforms and returns source estimate in the audio domain."""
        # Set data and estimate source.
        self.set_data(mix=audio)
        self.test()

        # Apply phase correction to estimate.
        phase_corrected = self.estimate_spec * torch.exp(1j * self.mix_phase)
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
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=100
        )
        self.optimizer.step()

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

        if delta > 0.01:
            self.stop_patience = self.training_params["stop_patience"]
            self._is_best_model = True
        else:
            self.stop_patience -= 1
            self._max_lr_steps -= 1 if not self.stop_patience else 0
            self._is_best_model = False
        return not self._max_lr_steps



    # @staticmethod
    # def copy_params(src_module):
    #     params_dest = {}
    #     for name, param in src_module.named_parameters():
    #         params_dest[name] = copy.deepcopy(param.data)
    #         param = param.to(dtype=torch.float16)
    #     return params_dest
    #
    # def update_f32_gradients(self):
    #     for name, param in self.model.named_parameters():
    #         if param.grad is not None:
    #             self.f32_weights[name].grad = param.grad.to(
    #                 dtype=torch.float32
    #             )
    #             self.f32_weights[name].grad /= self.scale
    #     self.model._parameters = self.f32_weights