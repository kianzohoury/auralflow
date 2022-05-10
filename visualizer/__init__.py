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
    audio_data: OrderedDict[str, Tensor],
    sample_rate: int = 44100,
) -> None:
    """Creates spectrogram images to visualize via tensorboard."""
    _, n_channels, n_bins, n_frames, n_targets = audio_data['mixture'].shape
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=512,
        sample_rate=sample_rate,
        f_max=sample_rate // 2,
        n_stft=n_bins
    )
    for name, audio_tensor in audio_data.items():
        mono_sample = torch.mean(audio_tensor[0], dim=0).detach().cpu()
        audio_data[name] = mono_sample.reshape((n_bins, n_frames, n_targets))

    # Crop high end frequency bins. 
    # n_bins = n_bins * 16384 // sample_rate // 2

    fig, ax = plt.subplots(dpi=900)

    # Log normalize spectrograms for better visualization.
    for i in range(n_targets):
        j = 0
        for name, audio_tensor in audio_data.items():
            mel_from_power = librosa.power_to_db(audio_tensor[:, :, i])
            # log_normalized = 20 * np.log10(audio_tensor[:, :, i] / np.max(audio_tensor[:, :, i]))
            # display.specshow(log_normalized, sr=44100, y_axis="log", ax=ax)
            ax.imshow(
                mel_from_power[:n_bins],
                origin="lower",
                extent=[0, 2, 1, 16384],
                aspect="auto",
                cmap='inferno'
            )

            format_plot(ax, f"{name}")
            
            break
            # format_plot(ax[i + j], f"{name}")
            j += 1

    plt.xlabel("Seconds")
    # fig.tight_layout()
    writer.add_figure("spectrograms", figure=fig, global_step=global_step)
    fig.savefig(f"{writer.log_dir}/spectrogram_step_{global_step}.png")
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

