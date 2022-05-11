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
    save_images: bool = False
) -> None:
    """Creates spectrogram images to visualize via tensorboard."""
    n_frames = min(estimate_audio.shape[-2], target_audio.shape[-2])
    n_bins = estimate_spec.shape[3]

    for i, label in enumerate(target_labels):
        fig, ax = plt.subplots(
            nrows=3, figsize=(12, 8), sharex=False, sharey=False, dpi=200
        )

        image = ax[0].imshow(
            torch.mean(target_spec, dim=1)[0, :, :, i].cpu(),
            origin="lower",
            aspect="auto",
            cmap='inferno'
        )
        image = ax[1].imshow(
            torch.mean(estimate_spec, dim=1)[0].cpu(),
            origin="lower",
            aspect="auto",
            cmap='inferno'
        )

        ax[2].set_facecolor('black')
        ax[2].plot(
            torch.mean(target_audio[0, :, :n_frames, i], dim=0).cpu(),
            color="yellowgreen",
            alpha=0.7,
            linewidth=0.2,
            label=f"{label} true"
        )
        ax[2].plot(
            torch.mean(estimate_audio[0, :, :n_frames, i], dim=0).cpu(),
            color="darkorange",
            alpha=0.7,
            linewidth=0.2,
            label=f"{label} estimate",
        )

        ax[0].set_title(f"{label} true")
        ax[1].set_title(f"{label} estimate")
        ax[2].set_title(f"{label} waveform")
        ax[2].set_xlim(xmin=0, xmax=n_frames)

        ax[0].set(frame_on=False)
        ax[1].set(frame_on=False)

        # fig.text(
        #     0.06,
        #     0.5,
        #     'Frequency bins',
        #     ha='center',
        #     va='center',
        #     rotation='vertical'
        # )

        format_plot(ax[0])
        format_plot(ax[1])


        # ax[0].set_ylabel

        plt.xlabel("Frames")
        plt.tight_layout()

        # Set the legend.
        legend = plt.legend(loc="upper right", framealpha=0)
        for leg in legend.legendHandles:
            leg.set_linewidth(3.0)
        for text in legend.get_texts():
            plt.setp(text, color = 'w')
        
        cbar = fig.colorbar(image, ax=ax.ravel())
        cbar.outline.set_visible(False)

        # Send figure to tensorboard and optionally save locally.
        writer.add_figure("spectrogram", figure=fig, global_step=global_step)
        if save_images:
            image_dir = Path(f"{writer.log_dir}/spectrogram_images")
            image_dir.mkdir(exist_ok=True)
            fig.savefig(image_dir / f"{label}_{global_step}.png")


def log_audio(
    writer: SummaryWriter,
    global_step: int,
    estimate_data: Tensor,
    target_data: Tensor,
    target_labels: List[str],
    sample_rate: int = 44100,
) -> None:
    """Logs audio data for listening via tensorboard."""
    n_batch, n_channels, n_frames, n_targets = estimate_data.shape
    target_data = target_data[:, :, :n_frames, :]

    # Collapse channel dimensions to mono and reshape.
    estimate_data = torch.mean(estimate_data, dim=1, keepdim=True)[0].reshape(
        (n_channels, n_frames, n_targets)
    )
    target_data = torch.mean(target_data, dim=1, keepdim=True)[0].reshape(
        (n_channels, n_frames, n_targets)
    )

    for i in range(len(target_labels)):
        writer.add_audio(
            tag=f"{target_labels[i]} estimate",
            snd_tensor=estimate_data[:, :, i].squeeze(-1),
            global_step=global_step,
            sample_rate=sample_rate,
        )
        writer.add_audio(
            tag=f"{target_labels[i]} true",
            snd_tensor=target_data[:, :, i].squeeze(-1),
            global_step=global_step,
            sample_rate=sample_rate,
        )


def format_plot(axis):
    """Helper plot formatting method."""
    # axis.set_yscale("log", basey=2)
    # axis.set_ylabel(tag)
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)
    axis.tick_params(axis="both", which="both", length=0)


def log_gradients(model: nn.Module, writer: SummaryWriter, global_step: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = torch.norm(param, 2)
            grad_norm = torch.norm(param.grad, 2)
            writer.add_histogram(f"{name}_grad_norm", grad_norm, global_step)
            writer.add_histogram(f"{name}_weight_norm", param_norm, global_step)

