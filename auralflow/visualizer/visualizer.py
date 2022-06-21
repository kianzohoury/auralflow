# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import matplotlib.pyplot as plt
import torch


from auralflow.models import SeparationModel
from auralflow.transforms import trim_audio
from matplotlib.figure import Figure
from pathlib import Path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union


def spec_show_diff(
    label: str, estimate: Tensor, target: Tensor, resolution: int = 120
) -> Figure:
    """Creates a ``pyplot`` figure comparing estimate and target spectrograms.

    The spectrogram subfigures are stacked vertically, with time as the
    horizontal axis, frequency as the vertical axis and amplitude as the
    intensity of the color map. Note that the channels are averaged and
    collapsed.

    Args:
        label (str): Target label.
        estimate (Tensor): Estimated target source spectrogram.
        target (Tensor): Ground-truth target source spectrogram.
        resolution (int): DPI.

    Returns:
        Figure: Spectrogram figure.

    :shape:
        - spectrogram: :math:`(C, F, T)`, non-batched

        where
            - :math:`C`: number of channels
            - :math:`F`: number of frequency bins
            - :math:`T`: number of time frames

    Examples:
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # get estimate and target spectrograms
        >>> estimate_spec = torch.rand((1, 512, 173))
        >>> target_spec = torch.rand((1, 512, 173))
        >>>
        >>> # generate figure
        >>> fig = spec_show_diff(
        ...     label="vocals",
        ...     estimate=estimate_spec,
        ...     target=target_spec,
        ...     resolution=120
        ... )
        >>>
        >>> # display figure
        >>> plt.show(fig)
        >>>
        >>> # save figure
        >>> plt.savefig(fname="spec_comparison.png")
    """
    fig, ax = plt.subplots(nrows=2, figsize=(16, 8), dpi=resolution)
    cmap = "inferno"

    ax[0].imshow(estimate, origin="lower", aspect="auto", cmap=cmap)
    img = ax[1].imshow(target, origin="lower", aspect="auto", cmap=cmap)

    # Formatting.
    ax[0].set_title(f"{label} estimate")
    ax[1].set_title(f"{label} true")
    ax[0].set(frame_on=False)
    ax[1].set(frame_on=False)

    _remove_ticks(ax)
    plt.xlabel("Frames")
    fig.supylabel("Frequency bins")
    plt.tight_layout()

    # Decibel color map.
    cbar = fig.colorbar(img, ax=ax.ravel())
    cbar.outline.set_visible(False)
    return fig


def waveform_show_diff(
    label: str, estimate: Tensor, target: Tensor, resolution: int = 120
) -> Figure:
    """Creates a ``pyplot`` figure comparing estimate and target waveforms.

    The waveforms are overlayed together, with time as the horizontal axis
    and amplitude as the vertical axis. Like ``spec_show_diff``, the channels
    are averaged and collapsed.

    Args:
        label (str): Target label.
        estimate (Tensor): Estimated target source audio signal.
        target (Tensor): Ground-truth target source audio signal.
        resolution (int): DPI.

    Returns:
        Figure: Waveform figure.

    :shape:
        - waveform: :math:`(C, T)`, non-batched

        where
            - :math:`C`: number of channels
            - :math:`T`: number of samples

    Examples:
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # get estimate and target audio signals
        >>> estimate_audio = torch.rand((1, 88200))
        >>> target_audio = torch.rand((1, 88200))
        >>>
        >>> # generate figure
        >>> fig = spec_show_diff(
        ...     label="vocals",
        ...     estimate=estimate_audio,
        ...     target=target_audio,
        ...     resolution=120
        ... )
        >>>
        >>> # display figure
        >>> plt.show(fig)
        >>>
        >>> # save figure
        >>> plt.savefig(fname="waveform_comparison.png")
    """
    fig, ax = plt.subplots(nrows=3, figsize=(12, 8), dpi=resolution)
    c1, c2 = "yellowgreen", "darkorange"
    est_label, target_label = f"{label} estimate", f"{label} target"

    for axis in ax:
        axis.set_facecolor("black")

    ax[0].plot(estimate, color=c1, linewidth=0.2)
    ax[1].plot(target, color=c2, linewidth=0.2)
    ax[2].plot(estimate, color=c1, alpha=0.7, linewidth=0.2, label=est_label)
    ax[2].plot(target, color=c2, alpha=0.5, linewidth=0.2, label=target_label)

    # Formatting.
    ax[0].set_title(f"{label} estimate")
    ax[1].set_title(f"{label} true")
    ax[0].set_xlim(xmin=0, xmax=estimate.shape[0])
    ax[1].set_xlim(xmin=0, xmax=estimate.shape[0])
    ax[2].set_xlim(xmin=0, xmax=estimate.shape[0])

    _remove_ticks(ax)
    plt.xlabel("Frames")
    fig.supylabel("Amplitude")
    plt.tight_layout()

    # Set the legend.
    figure_legend = plt.legend(loc="upper right", framealpha=0)
    for legend in figure_legend.legendHandles:
        legend.set_linewidth(3.0)
    for text in figure_legend.get_texts():
        plt.setp(text, color="w")
    return fig


def _remove_ticks(axes):
    """Helper method that formats axes by removing x/y-ticks."""
    for axis in axes:
        plt.setp(axis.get_xticklabels(), visible=False)
        plt.setp(axis.get_yticklabels(), visible=False)
        axis.tick_params(axis="both", which="both", length=0)


class TrainingVisualizer(object):
    """Visualizer that monitors training through images and audio playback.

    Especially useful for writing images and audio to a Tensorboard
    ``SummaryWriter``. Stores images under ``outer_dir/images`` and
    audio under ``output_dir/audio``.

    Args:
        output_dir (str): Outer directory to save output files.
        writer (Optional[SummaryWriter]): Tensorboard writer.
        view_spectrogram (bool): If True, calls ``show_spec_diff`` and writes
            the figure to tensorboard.
        view_waveform (bool): If True, calls ``show_waveform_diff`` and writes
            the figure to tensorboard.
        view_gradient (bool): If True, writes
            the figure to tensorboard.
    """

    spectrogram: dict
    audio: dict

    def __init__(
        self,
        output_dir: str,
        writer: Optional[SummaryWriter] = None,
        view_spectrogram: bool = True,
        view_waveform: bool = True,
        view_gradient: bool = True,
        play_audio: bool = True,
        num_images: int = 1,
        save_images: bool = False,
        save_audio: bool = False,
        interval: int = 10,
        sample_rate: int = 44100,
    ):
        super(TrainingVisualizer, self).__init__()
        self.writer = writer
        self.output_dir = output_dir
        self.view_spectrogram = view_spectrogram
        self.view_waveform = view_waveform
        self.view_gradient = view_gradient
        self.play_audio = play_audio
        self.num_images = num_images
        self.save_images = save_images
        self.save_audio = save_audio
        self.interval = interval
        self.sample_rate = sample_rate
        self._save_count = 0

    def _test_model(
        self,
        model: SeparationModel,
        mixture_audio: Tensor,
        target_audio: Tensor
    ) -> None:
        """Runs inference given a model, mixture and target audio data."""
        # Separate target source(s).
        estimate_audio = model.separate(mixture_audio)

        # Match audio lengths.
        mixture_audio, estimate_audio, target_audio = trim_audio(
            [mixture_audio, estimate_audio, target_audio]
        )

        # Apply log and mel scaling to estimate and target.
        estimate_mel = model.transform.to_mel_scale(model.estimate)
        target_mel = model.transform.audio_to_mel(target_audio)

        n_frames = estimate_audio.shape[-1]
        n_images = min(self.num_images, mixture_audio.shape[0])

        # Prepare audio for plotting.
        estimate_mel = torch.mean(estimate_mel, dim=1)[:n_images]
        target_mel = torch.mean(target_mel, dim=1)[:n_images]
        estimate_wav = torch.mean(estimate_audio, dim=1)[:n_images, :n_frames]
        target_wav = torch.mean(target_audio, dim=1)[:n_images, :n_frames]

        # Prepare audio for listening.
        estimate_res_audio = mixture_audio[..., :n_frames] - estimate_audio
        target_res_audio = mixture_audio[..., :n_frames]
        target_res_audio = target_res_audio - target_audio[..., :n_frames]

        # Store spectrograms.
        self.spectrogram = {
            "estimate_mono": estimate_mel.cpu(),
            "target_mono": target_mel.cpu(),
        }

        # Store audio.
        self.audio = {
            "estimate_source": estimate_audio.cpu(),
            "target_source": target_audio.cpu(),
            "estimate_residual": estimate_res_audio.cpu(),
            "target_residual": target_res_audio.cpu(),
            "estimate_mono": estimate_wav.cpu(),
            "target_mono": target_wav.cpu(),
        }


    def visualize_spectrogram(self, label: str, global_step: int) -> None:
        """Logs spectrogram images to tensorboard."""
        for i in range(self.num_images):
            spec_fig = spec_show_diff(
                label=label,
                estimate=self.spectrogram["estimate_mono"][i],
                target=self.spectrogram["target_mono"][i],
            )

            # Send figure to tensorboard.
            self.writer.add_figure(
                f"spectrogram/{label}",
                figure=spec_fig,
                global_step=global_step,
            )

            # Save image locally.
            if self.save_images and (self._save_count + 1) % self.interval == 0:
                self.save_figure(
                    figure=spec_fig,
                    filename=f"{label}_{global_step}_spectrogram.png",
                )

    def visualize_waveform(self, label: str, global_step: int) -> None:
        """Logs waveform images to tensorboard."""
        for i in range(self.num_images):
            wav_fig = waveform_show_diff(
                label=label,
                estimate=self.audio["estimate_mono"][i],
                target=self.audio["target_mono"][i],
            )

            # Send figure to tensorboard.
            self.writer.add_figure(
                f"waveform/{label}", figure=wav_fig, global_step=global_step
            )

            # Save image locally.
            if self.save_images and (self._save_count + 1) % self.interval == 0:
                self.save_figure(
                    figure=wav_fig,
                    filename=f"{label}_{global_step}_waveform.png",
                )

    def embed_audio(self, label: str, global_step: int) -> None:
        """Logs audio to tensorboard."""
        # Embed source estimate.
        self.writer.add_audio(
            tag=f"{label}/estimate",
            snd_tensor=self.audio["estimate_source"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )
        # Embed true source.
        self.writer.add_audio(
            tag=f"{label}/true",
            snd_tensor=self.audio["target_source"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )
        # Embed residual estimate.
        self.writer.add_audio(
            tag=f"residual/estimate",
            snd_tensor=self.audio["estimate_residual"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )
        # Embed true residual.
        self.writer.add_audio(
            tag=f"residual/true",
            snd_tensor=self.audio["target_residual"][0].T,
            global_step=global_step,
            sample_rate=self.sample_rate,
        )

    def visualize_gradient(self, model, global_step: int) -> None:
        """Sends model weights and gradients to tensorboard."""
        if self.view_gradient:
            for name, param in model.model.named_parameters():
                if param.grad is not None:
                    # Monitor model updates by tracking their 2-norms.
                    weight_norm = torch.linalg.norm(param)
                    grad_norm = torch.linalg.norm(param.grad)

                    # Don't log abnormal gradients.
                    log_weight = (
                        not weight_norm.isnan().any()
                        and not weight_norm.isinf().any()
                    )
                    log_grad = (
                        not grad_norm.isnan().any()
                        and not grad_norm.isinf().any()
                    )

                    if log_weight:
                        self.writer.add_histogram(
                            f"{name}_norm", weight_norm, global_step
                        )
                    if log_grad:
                        self.writer.add_histogram(
                            f"{name}_grad_norm", grad_norm, global_step
                        )

    def visualize(
        self, model, mixture: Tensor, target: Tensor, global_step: int
    ) -> None:
        """Runs visualizer."""
        mixture, target = mixture.to(model.device), target.to(model.device)
        for i, label in enumerate(model.target_labels):
            # Run model.
            self._test_model(
                model=model,
                mixture_audio=mixture,
                target_audio=target[..., i],
            )
            # Visualize spectrograms.
            if self.view_spectrogram:
                self.visualize_spectrogram(
                    label=label, global_step=global_step
                )
            # Visualize waveforms.
            if self.view_waveform:
                self.visualize_waveform(label=label, global_step=global_step)

            # Play separated audio back.
            if self.play_audio:
                self.embed_audio(label=label, global_step=global_step)
        # Increment save counter.
        self._save_count += 1


        # Create image folder.
        if save_image:
            Path(self.output_dir).mkdir(exist_ok=True)