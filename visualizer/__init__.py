from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

LINE_WIDTH = 79


def visualize_audio(
    model,
    mixture_audio: Tensor,
    target_audio: Tensor,
    to_tensorboard: bool = True,
    save_images: bool = False,
    global_step: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """Creates spectrogram/waveform images to visualize."""
    # Separate target source(s).
    model.eval()
    estimate_audio = model.separate(
        mixture_audio.to(model.device)
    ).squeeze(0).unsqueeze(-1)

    # Apply log and mel scaling to estimate and target.
    estimate_log_mel = model.transform.to_mel_scale(
        model.estimate.squeeze(0), to_db=True
    )
    target_log_mel = model.transform.audio_to_mel(
        target_audio.permute(2, 0, 1).to(model.device)
    ).permute(1, 2, 3, 0)

    # Create a figure for each target separately.
    for i, label in enumerate(model.target_labels):
        fig, ax = plt.subplots(
            nrows=3, figsize=(12, 8), sharex=False, sharey=False, dpi=200
        )

        # Collapse channels to mono.
        n_frames = estimate_audio.shape[-2]
        est_spec_mono = torch.mean(estimate_log_mel, dim=0).cpu()
        target_spec_mono = torch.mean(target_log_mel, dim=0)[:, :, i].cpu()
        est_wav = torch.mean(estimate_audio, dim=0)[:n_frames, i].cpu()
        target_wav = torch.mean(target_audio, dim=0)[:n_frames, i].cpu()

        # Plot spectrograms.
        ax[0].imshow(
            est_spec_mono, origin="lower", aspect="auto", cmap="inferno"
        )
        image = ax[1].imshow(
            target_spec_mono, origin="lower", aspect="auto", cmap="inferno"
        )

        # Plot waveforms.
        ax[2].set_facecolor("black")
        ax[2].plot(
            est_wav,
            color="yellowgreen",
            alpha=0.7,
            linewidth=0.2,
            label=f"{label} estimate",
        )
        ax[2].plot(
            target_wav,
            color="darkorange",
            alpha=0.7,
            linewidth=0.2,
            label=f"{label} true",
        )

        # Formatting.
        ax[0].set_title(f"{label} estimate")
        ax[1].set_title(f"{label} true")
        ax[2].set_title(f"{label} waveform")
        ax[2].set_xlim(xmin=0, xmax=n_frames)
        ax[0].set(frame_on=False)
        ax[1].set(frame_on=False)
        format_axis(ax[0])
        format_axis(ax[1])
        plt.xlabel("Frames")
        plt.tight_layout()

        # Set the legend.
        legend = plt.legend(loc="upper right", framealpha=0)
        for leg in legend.legendHandles:
            leg.set_linewidth(3.0)
        for text in legend.get_texts():
            plt.setp(text, color="w")

        # Decibel color map.
        cbar = fig.colorbar(image, ax=ax.ravel())
        cbar.outline.set_visible(False)

        # Send figures to tensorboard.
        if to_tensorboard and writer is not None:
            writer.add_figure(
                "spectrogram", figure=fig, global_step=global_step
            )

        # Save figures as images to disk.
        if save_images:
            image_dir = Path(f"{writer.log_dir}/spectrogram_images")
            image_dir.mkdir(exist_ok=True)
            fig.savefig(image_dir / f"{label}_{global_step}.png")


def listen_audio(
    model,
    mixture_audio: Tensor,
    target_audio: Tensor,
    writer: SummaryWriter,
    global_step: int,
    residual: bool = True,
    sample_rate: int = 44100,
) -> None:
    """Embed audio to tensorboard or save audio to disk."""
    # Separate audio.
    model.eval()
    estimate_audio = model.separate(
        mixture_audio.to(model.device)
    ).squeeze(0).unsqueeze(-1).cpu()

    # Trim to match estimate length. 
    n_frames = estimate_audio.shape[1]
    target_audio = target_audio[:, :n_frames, :]

    # Unpack batch dimension.
    mixture_audio = mixture_audio.squeeze(0)[:, :n_frames, :]

    # Send audio to tensorboard.
    for i, label in enumerate(model.target_labels):
        writer.add_audio(
            tag=f"{label} estimate",
            snd_tensor=estimate_audio[:, :, i],
            global_step=global_step,
            sample_rate=sample_rate,
        )
        writer.add_audio(
            tag=f"{label} true",
            snd_tensor=target_audio[:, :, i],
            global_step=global_step,
            sample_rate=sample_rate,
        )
        if residual:
            res_est = mixture_audio[:, :, i] - estimate_audio[:, :n_frames, i]
            writer.add_audio(
                tag=f"{label} residual estimate",
                snd_tensor=res_est.squeeze(-1),
                global_step=global_step,
                sample_rate=sample_rate,
            )


def log_gradients(model: nn.Module, writer: SummaryWriter, global_step: int):
    """Sends model weights and gradients to tensorboard."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Monitor model updates by tracking their 2-norms.
            weight_norm = torch.linalg.norm(param)
            grad_norm = torch.linalg.norm(param.grad)
            writer.add_histogram(f"{name}_norm", weight_norm, global_step)
            writer.add_histogram(f"{name}_norm", grad_norm, global_step)


def format_axis(axis):
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)
    axis.tick_params(axis="both", which="both", length=0)

