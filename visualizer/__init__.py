from typing import List, OrderedDict, Mapping, Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def log_spectrograms(
    writer: SummaryWriter,
    global_step: int,
    audio_data: OrderedDict[str, Tensor],
    sample_rate: int = 44100,
) -> None:
    """Creates spectrogram images to visualize via tensorboard."""
    _, n_channels, n_bins, n_frames, n_targets = audio_data['mixture'].shape
    for name, audio_tensor in audio_data.items():
        audio_data[name] = (
            torch.mean(audio_tensor[0], dim=0)
            .reshape((n_bins, n_frames, n_targets))
            .detach()
            .cpu()
        )

    fig, ax = plt.subplots(
        nrows=n_targets * 3, ncols=1, figsize=(11, 8), dpi=300
    )

    # Log normalize spectrograms for better visualization.
    for i in range(n_targets):
        j = 0
        for name, audio_tensor in audio_data.items():
            log_normalized = librosa.amplitude_to_db(
                audio_tensor[:, :, i], ref=np.max
            )
            ax[i + j].imshow(
                log_normalized,
                origin="lower",
                extent=[0, 12, 1, sample_rate // 2],
                aspect="auto",
                cmap="plasma",
            )
            format_plot(ax[i + j], f"{name}")
            j += 1

    plt.xlabel("Seconds")
    fig.tight_layout()
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
    axis.set_yscale("log")
    axis.set_ylabel(tag)
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

