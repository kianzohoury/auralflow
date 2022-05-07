import io
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.image import imread
from numpy import array
from torch import Tensor
from typing import List
from PIL import Image
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def log_spectrograms(
    writer: SummaryWriter,
    global_step: int,
    estimate_data: Tensor,
    target_data: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
) -> None:
    """Creates spectrogram images to visualize via tensorboard."""
    n_batch, n_channels, n_bins, n_frames, n_targets = estimate_data.shape
    estimate_data = (
        torch.mean(estimate_data[0], dim=0)
        .reshape((n_bins, n_frames, n_targets))
        .detach()
        .cpu()
    )
    target_data = (
        torch.mean(target_data[0], dim=0)
        .reshape((n_bins, n_frames, n_targets))
        .detach()
        .cpu()
    )

    estimates_log_normal, targets_log_normal = [], []
    for i in range(n_targets):
        estimates_log_normal.append(
            librosa.amplitude_to_db(estimate_data[:, :, i], ref=np.max)
        )
        targets_log_normal.append(
            librosa.amplitude_to_db(target_data[:, :, i], ref=np.max)
        )

    fig, ax = plt.subplots(
        nrows=n_targets * 2, ncols=1, figsize=(6, 3), dpi=200
    )

    # image = None
    for i in range(n_targets):
        # residual = estimates_log_normal[i] - targets_log_normal[i]
        ax[i].imshow(
            estimates_log_normal[i],
            origin="lower",
            extent=[0, 12, 1, sample_rate // 2],
            aspect="auto",
            cmap="inferno",
        )
        format_plot(ax[i], f"{target_labels[i]}_estimate")

        ax[i + 1].imshow(
            targets_log_normal[i],
            origin="lower",
            extent=[0, 12, 1, sample_rate // 2],
            aspect="auto",
            cmap="inferno",
        )
        format_plot(ax[i + 1], f"{target_labels[i]}_true")

    plt.xlabel("Seconds")
    fig.tight_layout()
    # fig.colorbar(image, ax=ax.ravel().tolist(), format="%+2.f dB")
    writer.add_figure("spectrograms", figure=fig, global_step=global_step)
    # plt.close(fig)


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
    # print(estimate_data.shape, target_data.shape)
    # print(torch.mean(estimate_data, dim=2, keepdim=True)[0].shape)
    estimate_data = torch.mean(estimate_data, dim=2, keepdim=True)[0].reshape(
        (n_channels, n_frames, n_targets)
    )
    target_data = torch.mean(target_data, dim=2, keepdim=True)[0].reshape(
        (n_channels, n_frames, n_targets)
    )
    # print(estimate_data.shape, target_data.shape)

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
    axis.set_yscale("log")
    axis.set_ylabel(tag)
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)
    axis.tick_params(axis="both", which="both", length=0)
