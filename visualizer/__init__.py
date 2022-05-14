from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def visualize_audio(
    model,
    mixture_audio: Tensor,
    target_audio: Tensor,
    n_samples: int = 1,
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
    )
    # Apply log and mel scaling to estimate and target.
    estimate_log_mel = model.transform.to_mel_scale(
        model.estimate, to_db=True
    )
    target_log_mel = model.transform.audio_to_mel(
        target_audio.permute(0, 2, 1).to(model.device)
    ).permute(0, 2, 1)

    # Collapse channels to mono.
    n_frames = estimate_audio.shape[-2]
    est_spec_mono = torch.mean(estimate_log_mel, dim=1).cpu()
    target_spec_mono = torch.mean(target_log_mel, dim=1).cpu()
    est_wav = torch.mean(estimate_audio, dim=1)[:, :n_frames].cpu()
    target_wav = torch.mean(target_audio, dim=0)[:, :n_frames].cpu()

    # Create n_samples number of figures.
    for i in range(n_samples):
        fig, ax = plt.subplots(
            nrows=3, figsize=(12, 8), sharex=False, sharey=False, dpi=120
        )

        # Plot spectrograms.
        ax[0].imshow(
            est_spec_mono[i], origin="lower", aspect="auto", cmap="inferno"
        )
        image = ax[1].imshow(
            target_spec_mono[i], origin="lower", aspect="auto", cmap="inferno"
        )

        # Plot waveforms.
        ax[2].set_facecolor("black")
        ax[2].plot(
            est_wav[i],
            color="yellowgreen",
            alpha=0.7,
            linewidth=0.2,
            label=f"{model.target_label} estimate",
        )
        ax[2].plot(
            target_wav[i],
            color="darkorange",
            alpha=0.7,
            linewidth=0.2,
            label=f"{model.target_label} true",
        )

        # Formatting.
        ax[0].set_title(f"{model.target_label} estimate")
        ax[1].set_title(f"{model.target_label} true")
        ax[2].set_title(f"{model.target_label} waveform")
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
            fig.savefig(image_dir / f"{model.target_label}_{global_step}.png")


def listen_audio(
    model,
    mixture_audio: Tensor,
    target_audio: Tensor,
    writer: SummaryWriter,
    global_step: int,
    n_samples: int = 1,
    sample_rate: int = 44100,
) -> None:
    """Embed audio to tensorboard or save audio to disk."""
    # Separate audio.
    model.eval()
    estimate_audio = model.separate(
        mixture_audio.to(model.device)
    ).cpu()

    # Trim target to match estimate length.
    n_frames = estimate_audio.shape[-1]
    target_audio = target_audio[:, :, :n_frames]
    mixture_audio = mixture_audio[:, :, :n_frames]

    # Send audio to tensorboard.
    for i in range(n_samples):
        writer.add_audio(
            tag=f"{model.target_label} estimate",
            snd_tensor=estimate_audio[i],
            global_step=global_step,
            sample_rate=sample_rate,
        )
        writer.add_audio(
            tag=f"{model.target_label} true",
            snd_tensor=target_audio[i],
            global_step=global_step,
            sample_rate=sample_rate,
        )
        residual_estimate = mixture_audio[i] - estimate_audio[i]
        writer.add_audio(
            tag=f"{model.target_label} residual estimate",
            snd_tensor=residual_estimate,
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

