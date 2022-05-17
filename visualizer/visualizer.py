from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def make_spectrogram_figure(
    label: str, estimate: Tensor, target: Tensor, resolution: int = 120
) -> matplotlib.figure:
    """Returns a spectrogram image as a matplotlib figure."""
    fig, ax = plt.subplots(nrows=2, figsize=(16, 8), dpi=resolution)
    ax[0].imshow(estimate, origin="lower", aspect="auto", cmap="inferno")
    img = ax[1].imshow(target, origin="lower", aspect="auto", cmap="inferno")

    # Formatting.
    ax[0].set_title(f"{label} estimate")
    ax[1].set_title(f"{label} true")
    ax[0].set(frame_on=False)
    ax[1].set(frame_on=False)
    _format_axes(ax)
    plt.xlabel("Frames")
    fig.supylabel("Frequency bins")
    plt.tight_layout()

    # Decibel color map.
    cbar = fig.colorbar(img, ax=ax.ravel())
    cbar.outline.set_visible(False)
    return fig


def make_waveform_figure(
    label: str, estimate: Tensor, target: Tensor, resolution: int = 120
) -> matplotlib.figure:
    """Returns a waveform image as a matplotlib figure."""
    fig, ax = plt.subplots(nrows=3, figsize=(12, 8), dpi=resolution)

    for axis in ax:
        axis.set_facecolor("black")

    ax[0].plot(estimate, color="yellowgreen", linewidth=0.2)
    ax[1].plot(target, color="darkorange", linewidth=0.2)
    ax[2].plot(
        estimate,
        color="yellowgreen",
        alpha=0.7,
        linewidth=0.2,
        label=f"{label} estimate",
    )
    ax[2].plot(
        target,
        color="darkorange",
        alpha=0.5,
        linewidth=0.2,
        label=f"{label} target",
    )

    # Formatting.
    ax[0].set_title(f"{label} estimate")
    ax[1].set_title(f"{label} true")
    ax[0].set_xlim(xmin=0, xmax=estimate.shape[0])
    ax[1].set_xlim(xmin=0, xmax=estimate.shape[0])
    ax[2].set_xlim(xmin=0, xmax=estimate.shape[0])
    _format_axes(ax)
    plt.xlabel("Frames")
    fig.supylabel("Amplitude")
    plt.tight_layout()

    # Set the legend.
    legend = plt.legend(loc="upper right", framealpha=0)
    for leg in legend.legendHandles:
        leg.set_linewidth(3.0)
    for text in legend.get_texts():
        plt.setp(text, color="w")
    return fig


def _format_axes(axes) -> None:
    for axis in axes:
        plt.setp(axis.get_xticklabels(), visible=False)
        plt.setp(axis.get_yticklabels(), visible=False)
        axis.tick_params(axis="both", which="both", length=0)


class Visualizer(object):
    """Wrapper class for visualizing/listening to results during training."""

    spectrogram: dict
    audio: dict

    def __init__(
        self,
        writer: SummaryWriter,
        save_dir: Union[str, Path],
        view_images: bool = True,
        view_gradients: bool = True,
        play_audio: bool = True,
        num_images: int = 1,
        save_image: bool = False,
        save_audio: bool = False,
        save_freq: int = 10,
        sample_rate: int = 44100,
    ):
        super(Visualizer, self).__init__()
        self.writer = writer
        self.save_dir = save_dir
        self.view_images = view_images
        self.view_gradients = view_gradients
        self.play_audio = play_audio
        self.num_images = num_images
        self.save_image = save_image
        self.save_audio = save_audio
        self.save_freq = save_freq
        self.sample_rate = sample_rate
        self.save_count = 0

        # Create image folder.
        if save_image:
            Path(self.save_dir).mkdir(exist_ok=True)

    def test_model(
        self, model, mixture_audio: Tensor, target_audio: Tensor
    ) -> None:
        """Runs model in inference mode and stores resulting audio tensors."""
        # Separate target source(s).
        estimate_audio = model.separate(mixture_audio)

        # Apply log and mel scaling to estimate and target.
        estimate_mel = model.transform.to_mel_scale(model.estimate)
        target_mel = model.transform.audio_to_mel(target_audio)

        n_frames = estimate_audio.shape[-1]
        n_images = min(self.num_images, mixture_audio.shape[0])

        # Collapse channels to mono.
        estimate_mel = torch.mean(estimate_mel, dim=1)[:n_images]
        target_mel = torch.mean(target_mel, dim=1)[:n_images]
        estimate_wav = torch.mean(estimate_audio, dim=1)[:n_images, :n_frames]
        target_wav = torch.mean(target_audio, dim=1)[:n_images, :n_frames]
        estimate_res_audio = mixture_audio[:, :, :n_frames] - estimate_audio

        # Store spectrograms.
        self.spectrogram = {
            "estimate_mono": estimate_mel.cpu(),
            "target_mono": target_mel.cpu(),
        }

        # Store audio.
        self.audio = {
            "estimate": estimate_audio.cpu(),
            "target": target_audio.cpu(),
            "residual": estimate_res_audio.cpu(),
            "estimate_mono": estimate_wav.cpu(),
            "target_mono": target_wav.cpu(),
        }

    def save_figure(self, figure: matplotlib.figure, filename: str) -> None:
        """Saves a matplotlib figure as an image."""
        figure.savefig(self.save_dir + "/" + filename, dpi=600)

    def visualize_spectrogram(self, label: str, global_step: int) -> None:
        """Logs spectrogram images to tensorboard."""
        for i in range(self.num_images):
            spec_fig = make_spectrogram_figure(
                label=label,
                estimate=self.spectrogram["estimate_mono"][i],
                target=self.spectrogram["target_mono"][i],
            )

            # Send figure to tensorboard.
            self.writer.add_figure(
                f"spectrogram/{label}", figure=spec_fig, global_step=global_step
            )

            # Save image locally.
            if self.save_image and (self.save_count + 1) % self.save_freq == 0:
                self.save_figure(
                    figure=spec_fig,
                    filename=f"{label}_{global_step}_spectrogram.png",
                )

    def visualize_waveform(self, label: str, global_step: int) -> None:
        """Logs waveform images to tensorboard."""
        for i in range(self.num_images):
            wav_fig = make_waveform_figure(
                label=label,
                estimate=self.audio["estimate_mono"][i],
                target=self.audio["target_mono"][i],
            )

            # Send figure to tensorboard.
            self.writer.add_figure(
                f"waveform/{label}", figure=wav_fig, global_step=global_step
            )

            # Save image locally.
            if self.save_image and (self.save_count + 1) % self.save_freq == 0:
                self.save_figure(
                    figure=wav_fig,
                    filename=f"{label}_{global_step}_waveform.png",
                )

    def embed_audio(self, label: str, global_step: int) -> None:
        """Logs audio to tensorboard."""
        # Send audio to tensorboard.
        self.writer.add_audio(
            tag=f"{label}/estimate",
            snd_tensor=self.audio["estimate"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )
        self.writer.add_audio(
            tag=f"{label}/true",
            snd_tensor=self.audio["target"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )
        self.writer.add_audio(
            tag=f"{label}/estimate_residual",
            snd_tensor=self.audio["residual"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )

    def visualize_gradient(self, model, global_step: int) -> None:
        """Sends model weights and gradients to tensorboard."""
        if self.view_gradients:
            for name, param in model.model.named_parameters():
                if param.grad is not None:
                    # Monitor model updates by tracking their 2-norms.
                    weight_norm = torch.linalg.norm(param)
                    grad_norm = torch.linalg.norm(param.grad)
                    self.writer.add_histogram(
                        f"{name}_norm", weight_norm, global_step
                    )
                    self.writer.add_histogram(
                        f"{name}_grad_norm", grad_norm, global_step
                    )

    def visualize(
        self, model, mixture: Tensor, target: Tensor, global_step: int
    ) -> None:
        """Runs visualizer."""
        for i, label in enumerate(model.target_labels):
            # Run model.
            self.test_model(
                model=model, 
                mixture_audio=mixture[:, :, :, i].to(model.device),
                target_audio=target[:, :, :, i].to(model.device)
            )

            # Visualize images.
            if self.view_images:
                self.visualize_spectrogram(
                    label=label, global_step=global_step
                )
                self.visualize_waveform(label=label, global_step=global_step)

            # Play separated audio back.
            if self.play_audio:
                self.embed_audio(label=label, global_step=global_step)

        # Increment save counter.
        self.save_count += 1
