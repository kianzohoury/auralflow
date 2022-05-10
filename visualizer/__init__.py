from typing import List, OrderedDict, Mapping, Dict

import librosa
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torchaudio

from librosa import display


def log_spectrograms(
    writer: SummaryWriter,
    global_step: int,
    estimate_spec: Tensor,
    target_spec: Tensor,
    estimate_audio: Tensor,
    target_audio: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
) -> None:
    """Creates spectrogram images to visualize via tensorboard."""

    for i, label in enumerate(target_labels):
        fig, ax = plt.subplots(
            nrows=3, figsize=(12, 12), sharex=False, dpi=900
        )
        ax[0].imshow(
            torch.mean(target_spec, dim=0),
            origin="lower",
            aspect="auto",
            cmap='viridis'
        )
        ax[1].imshow(
            torch.mean(estimate_spec, dim=0),
            origin="lower",
            aspect="auto",
            cmap='inferno'
        )
        ax[2].set_facecolor('black')
        ax[2].plot(
            target_audio, color="yellowgreen", alpha=0.7, linewidth=0.2
        )
        ax[2].plot(
            estimate_audio, color="darkorange", alpha=0.7, linewidth=0.2
        )
        ax[2].set_xlim(xmin=0, xmax=target_audio.shape[-1])
        plt.xlabel("Frames")
        fig.tight_layout()
        writer.add_figure("spectrogram", figure=fig, global_step=global_step)
        fig.savefig(
            f"{writer.log_dir}/spectrogram_step_{global_step}.png"
        )


def log_audio(
    writer: SummaryWriter,
    global_step: int,
    estimate_data: Tensor,
    target_data: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
) -> None:
    """Logs audio data for listening via tensorboard."""
    n_batch, n_frames, n_channels, n_targets = estimate_data.shape
    target_data = target_data[:, :n_frames, :, :]

    # Collapse channel dimensions to mono and reshape.
    estimate_data = torch.mean(estimate_data, dim=2, keepdim=True)[0].reshape(
        (n_channels, n_frames, n_targets)
    )
    target_data = torch.mean(target_data, dim=2, keepdim=True)[0].reshape(
        (n_channels, n_frames, n_targets)
    )

    for i in range(len(target_labels)):
        writer.add_audio(
            tag=f"{target_labels[i]}_estimate",
            snd_tensor=estimate_data[:, :, i].squeeze(-1),
            global_step=global_step,
            sample_rate=sample_rate,
        )
        writer.add_audio(
            tag=f"{target_labels[i]}_true",
            snd_tensor=target_data[:, :, i].squeeze(-1),
            global_step=global_step,
            sample_rate=sample_rate,
        )


def format_plot(axis, tag):
    """Helper plot formatting method."""
    axis.set_yscale("log", basey=2)
    axis.set_ylabel(tag)
    # plt.setp(axis.get_xticklabels(), visible=False)
    # plt.setp(axis.get_yticklabels(), visible=False)
    # axis.tick_params(axis="both", which="both", length=0)


def log_gradients(model: nn.Module, writer: SummaryWriter, global_step: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = torch.norm(param, 2)
            grad_norm = torch.norm(param.grad, 2)
            writer.add_histogram(f"{name}_grad_norm", grad_norm, global_step)
            writer.add_histogram(f"{name}_weight_norm", param_norm, global_step)

