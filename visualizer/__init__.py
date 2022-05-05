import io
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.image import imread
from numpy import array
from torch import Tensor
from typing import List


def get_residual_specs_image(
    estimate_data: Tensor,
    target_data: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
) -> array:
    """Creates residual spectrogram images for diisplaying via tensorboard."""

    n_batch, n_channels, n_bins, n_frames, n_targets = estimate_data.shape
    estimate_data, target_data = estimate_data.cpu(), target_data.cpu()
    estimate_data = torch.mean(estimate_data, dim=1).reshape(
        (0, n_bins, n_frames, n_targets)
    )
    target_data = torch.mean(target_data, dim=1).reshape(
        (0, n_bins, n_frames, n_targets)
    )

    estimates_log_normal, targets_log_normal = [], []
    for i in range(n_targets):
        estimates_log_normal[i] = librosa.amplitude_to_db(
            estimate_data[i], ref=np.max
        )
        targets_log_normal[i] = librosa.amplitude_to_db(
            target_data[i], ref=np.max
        )

    fig, ax = plt.subplots(nrows=n_targets, ncols=1, figsize=(12, 6), dpi=150)

    image = None
    for i in range(n_targets):
        residual = estimates_log_normal[i] - targets_log_normal[i]
        image = ax[i].imshow(
            residual,
            origin="lower",
            extent=[0, 12, 1, sample_rate // 2],
            aspect="auto",
        )
        format_plot(ax[i], target_labels[i])

    plt.suptitle("Residual spectrograms")
    plt.xlabel("Seconds")
    fig.tight_layout()
    fig.colorbar(image, ax=ax.ravel().tolist(), format="%+2.f dB")

    buffer = io.BytesIO()
    plt.imsave(buffer, format="jpg")
    image = imread(buffer)
    return image


def format_plot(axis, target_label):
    """Helper plot formatting method."""
    axis.set_yscale("log")
    axis.set_ylabel(f"residual {target_label}")
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)
    axis.tick_params(axis="both", which="both", length=0)
