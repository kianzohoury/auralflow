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


def log_residual_specs(
    writer: SummaryWriter,
    global_step: int,
    estimate_data: Tensor,
    target_data: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
):
    """Plots residual spectrograms to tensorboard."""
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
        nrows=n_targets + 1, ncols=1, figsize=(6, 3), dpi=200
    )

    image = None
    for i in range(n_targets):
        estimates_log_normal[i][estimates_log_normal[i] <= -70] = np.nan
        residual = estimates_log_normal[i] - targets_log_normal[i]
        image = ax[i].imshow(
            estimates_log_normal[i],
            origin="lower",
            extent=[0, 12, 1, sample_rate // 2],
            aspect="auto",
            cmap="Reds",
            alpha=0.8
        )
        targets_log_normal[i][targets_log_normal[i] <= -70] = np.nan
        image = ax[i].imshow(
            targets_log_normal[i],
            origin="lower",
            extent=[0, 12, 1, sample_rate // 2],
            aspect="auto",
            cmap="Blues",
            alpha=0.8
        )
        format_plot(ax[i], target_labels[i])

    plt.xlabel("Seconds")
    fig.tight_layout()
    fig.colorbar(image, ax=ax.ravel().tolist(), format="%+2.f dB")

    writer.add_figure("residual_specs", figure=fig, global_step=global_step)
    # plt.close(fig)


def format_plot(axis, target_label):
    """Helper plot formatting method."""
    axis.set_yscale("log")
    axis.set_ylabel(f"residual {target_label}")
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)
    axis.tick_params(axis="both", which="both", length=0)
