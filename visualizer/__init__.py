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


def get_residual_specs_image(
    estimate_data: Tensor,
    target_data: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
):
    """Creates residual spectrogram images for displaying via tensorboard."""
    n_batch, n_channels, n_bins, n_frames, n_targets = estimate_data.shape
    estimate_data = torch.mean(estimate_data[0], dim=0).reshape(
        (n_bins, n_frames, n_targets)
    ).detach().cpu()
    target_data = torch.mean(target_data[0], dim=0).reshape(
        (n_bins, n_frames, n_targets)
    ).detach().cpu()

    estimates_log_normal, targets_log_normal = [], []
    for i in range(n_targets):
        estimates_log_normal.append(
            librosa.amplitude_to_db(
                estimate_data[:, :, i], ref=np.max
            )
        )
        targets_log_normal.append(
            librosa.amplitude_to_db(
                target_data[:, :, i], ref=np.max
            )
        )

    fig, ax = plt.subplots(nrows=n_targets + 1, ncols=1, figsize=(6, 3), dpi=150)

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

    with open("temp.png", "w") as temp_file:
        plt.savefig("temp.png", dpi=120)
        img_data = torch.from_numpy(np.array(Image.open("temp.png")))
        Path("temp.png").unlink()

    plt.close(fig)
    return img_data


def format_plot(axis, target_label):
    """Helper plot formatting method."""
    axis.set_yscale("log")
    axis.set_ylabel(f"residual {target_label}")
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)
    axis.tick_params(axis="both", which="both", length=0)
