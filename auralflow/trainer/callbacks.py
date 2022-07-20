# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git
from abc import ABC, abstractmethod

import inspect
from pathlib import Path

import torch

from typing import Optional, Callable, List, Dict
from auralflow.visualizer import spec_show_diff, waveform_show_diff
from auralflow.transforms import trim_audio
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

from auralflow.losses import get_evaluation_metrics
from auralflow.models import SeparationModel, SpectrogramMaskModel
from auralflow.visualizer import TrainingVisualizer


class TrainingCallback(ABC):
    r"""Abstract base class for all training callbacks.

    This class should not be instantiated, but rather subclassed.

    Args:
        event (str): Event that triggers the callback: ``'on_iteration_end'`` |
            ``'on_loss_end'`` | ``'on_epoch_start'`` | ``'on_epoch_end'``.

    Examples:

        >>> from auralflow.trainer import TrainingCallback
        >>> class MyCustomCallback(TrainingCallback):
        ...     def __init__(self, event, ...):
        ...         super(MyCustomCallback, self).__init__(event)
        ...         # handle instance attributes
        ...         ...
        ...     def __call__(self, **kwargs):
        ...         # define callback here
        ...         ...
        >>> my_callback = MyCustomCallback(event="on_epoch_end", ...)
        >>> my_callback(...)
    """

    def __init__(self, event: str):
        self.event = event

    @abstractmethod
    def __call__(self, **kwargs):
        """
        All subclasses must implement this method.

        Keyword Args:
            kwargs: Keyword arguments.
        """
        pass


class CallbackManager(object):
    r"""Handles a group of training callbacks.

    By default, calling a method corresponding to an event will have no effect.
    Each event can be handled by either implementing the corresponding
    handler method manually, or by calling ``add_callback(...)``, which bins a
    ``TrainingCallback`` to its event.

    Note that this class currently has limited functionality and is not stable
    due to the current implementation of ``trainer.ModelTrainer``.

    Examples:

        >>> from auralflow.trainer import CallbackManager, LayersVisualCallback
        >>> callbacks = CallbackManager()
        >>> # create a new callback
        >>> layers_callback = LayersVisualCallback(...)
        >>> # add to callbacks manager
        >>> callbacks.add_callback(layers_callback)
    """

    _callbacks: Dict[str, List[TrainingCallback]]

    def __init__(self):
        self._callbacks = {}

    def on_iteration_end(self, global_step: int, **kwargs):
        """Post-iteration callback."""
        for callback in self._callbacks.get("on_iteration_end", []):
            callback(global_step=global_step, **kwargs)

    def on_loss_end(self, global_step: int, **kwargs):
        """Post-backward phase callback."""
        for callback in self._callbacks.get("on_loss_end", []):
            callback(global_step=global_step, **kwargs)

    def on_epoch_start(self, epoch: int, **kwargs):
        """Pre-epoch callback."""
        for callback in self._callbacks.get("on_epoch_start", []):
            callback(global_step=epoch, **kwargs)

    def on_epoch_end(self, epoch: int, **kwargs):
        """Post-epoch callback."""
        for callback in self._callbacks.get("on_epoch_end", []):
            callback(global_step=epoch, **kwargs)

    def add_callback(self, callback: TrainingCallback) -> None:
        """Attach a callback to the group associated with its event."""
        if callback.event not in self._callbacks:
            self._callbacks[callback.event] = []
        self._callbacks[callback.event].append(callback)


#
# class SeparationMetricCallback(TrainingCallback):
#     """Callback class for printing evaluation metrics."""
#
#     def __init__(self, model: SeparationModel, disp_freq: int = 5):
#         self.model = model
#         self.best_metrics = {
#             "pesq": float("-inf"),
#             "sar": float("-inf"),
#             "sdr": float("-inf"),
#             "si_sdr": float("-inf"),
#             "sir": float("inf"),
#             "stoi": float("-inf"),
#         }
#         self.disp_freq = disp_freq
#         self.count = 0
#
#     def on_epoch_end(self, mix: Tensor, target: Tensor) -> None:
#         """Prints source separation evaluation metrics at epoch finish."""
#         if (self.count + 1) % self.disp_freq != 0:
#             self.count += 1
#             return
#         estimate = self.model.separate(audio=mix)
#         print("Calculating evaluation metrics...")
#         metrics = get_evaluation_metrics(
#             mixture=mix, estimate=estimate, target=target
#         )
#         table = PrettyTable(["Metric", "Value"])
#         table.align = "l"
#         for metric_label, val in metrics.items():
#             # Skip SIR for now.
#             self.model.metrics[metric_label] = val
#             if metric_label == "sir":
#                 continue
#             if val > self.best_metrics[metric_label]:
#                 self.best_metrics[metric_label] = val
#                 val = str(val) + "*"
#             table.add_row([metric_label, val])
#         print(table)
#         self.count = 0


class LossCallback(TrainingCallback):
    """Losses callback."""
    def __init__(
        self,
        event: str,
        tensorboard_writer: SummaryWriter,
        main_tag: str = "loss/epoch"
    ) -> None:
        super(LossCallback, self).__init__(event=event)
        self._writer = tensorboard_writer
        self._main_tag = main_tag

    def __call__(
        self,
        named_losses: Dict[str, float],
        global_step: int,
        **kwargs
    ) -> None:
        self._writer.add_scalars(
            main_tag=self._main_tag,
            tag_scalar_dict=named_losses,
            global_step=global_step,
        )


def _visualize_layers_handler(
    tensorboard_writer: SummaryWriter,
    model: SeparationModel,
    global_step: int,
    show_weights: bool = True,
    show_grad: bool = True,
    use_norm: bool = True
) -> None:
    """Visualizes the model's layer weights/loss gradients during training.

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        global_step (int): Global step.
        show_weights (bool): If ``True``, logs the model's weights for each
            each layer. Default: ``True``.
        show_grad (bool): If ``True``, logs the gradients with
            respect to each layer. Default: ``True``.
        use_norm (bool): If ``True``, logs the 2-norms of the weights and
            gradients. Default: ``True``.
    """
    for name, param in model.model.named_parameters():
        # Only log trainable parameters.
        if param.grad is not None:
            # Log weights.
            weights = torch.linalg.norm(param) if use_norm else param
            if show_weights:
                if not (weights.isnan().any() or weights.isinf().any()):
                    tensorboard_writer.add_histogram(
                        f"{name}", weights, global_step
                    )
            # Log gradients.
            grad = torch.linalg.norm(param.grad) if use_norm else param.grad
            if show_grad:
                if not (grad.isnan().any() or grad.isinf().any()):
                    tensorboard_writer.add_histogram(
                        f"{name}_grad", weights, global_step
                    )


class LayersVisualCallback(TrainingCallback):
    """Visualizes model parameters by layer via tensorboard.

    Note that this callback is only triggered on ``on_loss_end`` events.

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        show_weights (bool): If ``True``, logs the model's weights for each
            each layer. Default: ``True``.
        show_grad (bool): If ``True``, logs the gradients with
            respect to each layer. Default: ``True``.
        use_norm (bool): If ``True``, logs the 2-norms of the weights and
            gradients. Default: ``True``.
    """

    def __init__(
        self,
        tensorboard_writer: SummaryWriter,
        model: SeparationModel,
        show_weights: bool = True,
        show_grad: bool = True,
        use_norm: bool = True,
    ):
        super(LayersVisualCallback, self).__init__(event="on_loss_end")
        self._writer = tensorboard_writer
        self._model = model
        self.show_weights = show_weights
        self.show_grad = show_grad
        self.use_norm = use_norm

    def __call__(self, global_step: int, **kwargs) -> None:
        """Calls the event handler for visualizing model layers.

        Args:
            global_step (int): Global step.

        """
        _visualize_layers_handler(
            tensorboard_writer=self._writer,
            model=self._model,
            global_step=global_step,
            show_weights=self.show_weights,
            show_grad=self.show_grad,
            use_norm=self.use_norm
        )


def _waveform_visual_handler(
    tensorboard_writer: SummaryWriter,
    model: SeparationModel,
    mixture_audio: Tensor,
    target_audio: Tensor,
    global_step: int,
    save_dir: Optional[str] = None
) -> None:
    """Creates waveform images for visualizing via tensorboard (and locally).

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        mixture_audio (Tensor): Mixture audio signal.
        target_audio (Tensor): Target audio signal.
        global_step (int): Global step.
        save_dir (optional[str]): Directory to save the image under. If
            no path is specified, the image is not saved. If no directory with
            that name exists, a new one is created. Default: ``None``.

    Raises:
        OSError: Raised if the image cannot be saved.
    """
    # Separate audio.
    estimate_audio = model.separate(mixture=mixture_audio)
    estimate_audio, mixture_audio = trim_audio([estimate_audio, mixture_audio])

    # Take only the first tensors from the batch.
    estimate_wav = torch.mean(estimate_audio, dim=1)[0]
    target_wav = torch.mean(target_audio, dim=1)[0]

    # Compensate for single-target models for now.
    if estimate_wav.dim() == 1:
        estimate_wav = estimate_wav.unsqueeze(-1).cpu()
    if target_wav.dim() == 1:
        target_wav = target_wav.unsqueeze(-1).cpu()

    # Create waveform figure for each target source.
    for i, label in enumerate(model.targets):
        wav_fig = waveform_show_diff(
            label=label,
            estimate=estimate_wav[..., i],
            target=target_wav[..., i]
        )

        # Send figure to tensorboard.
        tensorboard_writer.add_figure(
            f"waveform/{label}", figure=wav_fig, global_step=global_step
        )

        # Save image locally if specified.
        if save_dir is not None:
            # Create an empty directory if one does not already exist.
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            try:
                full_path = Path(
                    save_dir, f"{label}_{global_step}_waveform.png"
                ).absolute()
                wav_fig.savefig(fname=str(full_path))
            except OSError as error:
                raise error


class WaveformVisualCallback(TrainingCallback):
    r"""Visualizes waveform images via tensorboard (and locally).

    See :func:`~auralflow.visualizer.waveform_show_diff` for more info. Note
    that this callback is only triggered on ``'on_epoch_end'`` events.

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        save_dir (optional[str]): Directory to save the image under. If
            no path is specified, the image is not saved. If no directory with
            that name exists, a new one is created. Default: ``None``.
        save_freq (int): Frequency (in epochs) at which images are saved.
            Default: ``5``.
    """
    def __init__(
        self,
        tensorboard_writer: SummaryWriter,
        model: SeparationModel,
        save_dir: Optional[str] = None,
        save_freq: int = 5
    ):
        super(WaveformVisualCallback, self).__init__(event="on_epoch_end")
        self._writer = tensorboard_writer
        self._model = model
        self.save_dir = save_dir
        self.save_freq = save_freq
        self._count = 0

    def __call__(
        self,
        mixture_audio: Tensor,
        target_audio: Tensor,
        global_step: int,
        **kwargs
    ) -> None:
        """Calls the event handler for visualizing waveform images.

        Args:
            mixture_audio (Tensor): Mixture audio signal of dimension
                `(batch, channels, time)`.
            target_audio (Tensor): Target audio signal of dimension
                `(batch, channels, times)`.
            global_step (int): Global step.

        Raised:
            OSError: Raised if the image cannot be saved under
            ``self.save_dir``.
        """
        if (self._count + 1) % self.save_freq == 0:
            save_dir = self.save_dir
        else:
            save_dir = None
        self._count += 1

        _waveform_visual_handler(
            tensorboard_writer=self._writer,
            model=self._model,
            mixture_audio=mixture_audio,
            target_audio=target_audio,
            global_step=global_step,
            save_dir=save_dir
        )


def _spectrogram_visual_handler(
    tensorboard_writer: SummaryWriter,
    model: SpectrogramMaskModel,
    mixture_audio: Tensor,
    target_audio: Tensor,
    global_step: int,
    save_dir: Optional[str] = None
) -> None:
    """Creates waveform images for visualizing via tensorboard (and locally).

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        mixture_audio (Tensor): Mixture audio signal.
        target_audio (Tensor): Target audio signal.
        global_step (int): Global step.
        save_dir (optional[str]): Directory to save the image under. If
            no path is specified, the image is not saved. If no directory with
            that name exists, a new one is created. Default: ``None``.

    Raises:
        OSError: Raised if the image cannot be saved.
    """
    # Separate audio.
    estimate_audio = model.separate(mixture=mixture_audio)
    estimate_audio, mixture_audio = trim_audio([estimate_audio, mixture_audio])

    # Apply log and mel scaling to estimate and target spectrograms.
    estimate_mel = model.audio_transform.audio_to_mel(audio=estimate_audio)
    target_mel = model.audio_transform.audio_to_mel(audio=target_audio)

    # Take only the first tensors from the batch.
    estimate_mel = torch.mean(estimate_mel, dim=1)[0]
    target_mel = torch.mean(target_mel, dim=1)[0]

    # Compensate for single-target models for now.
    if estimate_mel.dim() == 3:
        estimate_mel = estimate_mel.unsqueeze(-1).cpu()
    if target_mel.dim() == 3:
        target_mel = target_mel.unsqueeze(-1).cpu()

    # Create spectrogram figure for each target source.
    for i, label in enumerate(model.targets):
        spec_fig = spec_show_diff(
            label=label,
            estimate=estimate_mel[..., i],
            target=target_mel[..., i]
        )

        # Send figure to tensorboard.
        tensorboard_writer.add_figure(
            f"spectrogram/{label}", figure=spec_fig, global_step=global_step
        )

        # Save image locally if specified.
        if save_dir is not None:
            # Create an empty directory if one does not already exist.
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            try:
                full_path = Path(
                    save_dir, f"{label}_{global_step}_spectrogram.png"
                ).absolute()
                spec_fig.savefig(fname=str(full_path))
            except OSError as error:
                raise error


class SpectrogramVisualCallback(TrainingCallback):
    r"""Visualizes spectrogram images via tensorboard (and locally).

    See :func:`~auralflow.visualizer.spec_show_diff` for more info. Note that
    this callback is only triggered on ``'on_epoch_end'`` events.

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        save_dir (optional[str]): Directory to save the image under. If
            no path is specified, the image is not saved. If no directory with
            that name exists, a new one is created. Default: ``None``.
        save_freq (int): Frequency (in epochs) at which images are saved.
            Default: ``5``.
    """
    def __init__(
        self,
        tensorboard_writer: SummaryWriter,
        model: SpectrogramMaskModel,
        save_dir: Optional[str] = None,
        save_freq: int = 5
    ):
        super(SpectrogramVisualCallback, self).__init__(event="on_epoch_end")
        self._writer = tensorboard_writer
        self._model = model
        self.save_dir = save_dir
        self.save_freq = save_freq
        self._count = 0

    def __call__(
        self,
        mixture_audio: Tensor,
        target_audio: Tensor,
        global_step: int,
        **kwargs
    ) -> None:
        """Calls the event handler for visualizing spectrogram images.

        Args:
            mixture_audio (Tensor): Mixture audio signal of dimension
                `(batch, channels, time)`.
            target_audio (Tensor): Target audio signal of dimension
                `(batch, channels, time)`.
            global_step (int): Global step.

        Raised:
            OSError: Raised if the image cannot be saved under
            ``self.save_dir``.
        """
        if (self._count + 1) % self.save_freq == 0:
            save_dir = self.save_dir
        else:
            save_dir = None
        self._count += 1

        _spectrogram_visual_handler(
            tensorboard_writer=self._writer,
            model=self._model,
            mixture_audio=mixture_audio,
            target_audio=target_audio,
            global_step=global_step,
            save_dir=save_dir
        )


def _audio_player_handler(
    tensorboard_writer: SummaryWriter,
    model: SeparationModel,
    mixture_audio: Tensor,
    target_audio: Tensor,
    global_step: int,
    embed_residual: bool = True,
    sample_rate: int = 44100
) -> None:
    """Logs audio to tensorboard."""
    # Separate audio.
    estimate_audio = model.separate(mixture=mixture_audio).unsqueeze(-1)
    print(estimate_audio.shape, target_audio.shape, mixture_audio.unsqueeze(-1).shape)
    estimate_audio, mixture_audio, target_audio = trim_audio(
        [estimate_audio, mixture_audio.unsqueeze(-1), target_audio]
    )
    print(estimate_audio.shape, target_audio.shape, mixture_audio.shape)

    # Compensate for single-target models for now. # Take only the first
    # tensors from the batch.
    if estimate_audio.dim() == 3:
        estimate_audio = estimate_audio.unsqueeze(-1)
    if target_audio.dim() == 3:
        target_audio = target_audio.unsqueeze(-1)

    for i, label in enumerate(model.targets):
        # Embed source estimate.
        print(estimate_audio.shape)
        tensorboard_writer.add_audio(
            tag=f"{label}/estimate",
            snd_tensor=estimate_audio[0, ..., i].T.cpu(),
            global_step=global_step,
            sample_rate=sample_rate,
        )
        # Embed true source.
        tensorboard_writer.add_audio(
            tag=f"{label}/true",
            snd_tensor=target_audio[0, ..., i].T.cpu(),
            global_step=global_step,
            sample_rate=sample_rate,
        )

        if embed_residual:
            estimate_res_audio = mixture_audio[0] - estimate_audio[..., i]
            target_res_audio = mixture_audio[0] - target_audio[..., i]

            # Embed residual estimate.
            tensorboard_writer.add_audio(
                tag=f"residual/estimate",
                snd_tensor=estimate_res_audio.T,
                global_step=global_step,
                sample_rate=sample_rate,
            )
            # Embed true residual.
            tensorboard_writer.add_audio(
                tag=f"residual/true",
                snd_tensor=target_res_audio.T,
                global_step=global_step,
                sample_rate=sample_rate,
            )


class AudioPlayerCallback(TrainingCallback):
    """Plays back separated audio via tensorboard.

    Note that this callback is only triggered on ``'on_epoch_end'`` events.

    Args:
        tensorboard_writer (SummaryWriter): Tensorboard writer.
        model (SeparationModel): Model.
        embed_residual (bool): Additionally embeds the residual (background)
            track. Default: ``True``.
        sample_rate (int): Sample rate. Default: ``44100``.
    """
    def __init__(
        self,
        tensorboard_writer: SummaryWriter,
        model: SeparationModel,
        embed_residual: bool = True,
        sample_rate: int = 44100
    ):
        super(AudioPlayerCallback, self).__init__(event="on_epoch_end")
        self._writer = tensorboard_writer
        self._model = model
        self._embed_residual = embed_residual
        self._sample_rate = sample_rate

    def __call__(
        self,
        mixture_audio: Tensor,
        target_audio: Tensor,
        global_step: int,
        **kwargs
    ) -> None:
        """Calls the event handler for playing back audio.

        Args:
            mixture_audio (Tensor): Mixture audio signal of dimension
                `(batch, channels, time)`.
            target_audio (Tensor): Target audio signal of dimension
                `(batch, channels, time)`.
            global_step (int): Global step.
        """
        _audio_player_handler(
            tensorboard_writer=self._writer,
            model=self._model,
            mixture_audio=mixture_audio,
            target_audio=target_audio,
            global_step=global_step,
            embed_residual=self._embed_residual,
            sample_rate=self._sample_rate
        )


def _create_callbacks(
    model: SeparationModel,
    tensorboard_writer: Optional[SummaryWriter],
    save_dir: Optional[str],
    save_freq: int = 5,
    write_iter_loss: bool = True,
    write_epoch_loss: bool = True,
    visualize_weights: bool = True,
    visualize_gradients: bool = True,
    visualize_norm: bool = True,
    visualize_waveform: bool = True,
    visualize_spectrogram: bool = True,
    play_audio: bool = True,
    embed_residual: bool = True,
    sample_rate: int = 44100
) -> CallbackManager:
    """Helper method that creates a ``CallbackManager`` for training."""
    callbacks = CallbackManager()
    if write_iter_loss:
        callbacks.add_callback(
            LossCallback(
                event="on_iteration_end",
                tensorboard_writer=tensorboard_writer,
                main_tag="loss/iter"
            )
        )
    if write_epoch_loss:
        callbacks.add_callback(
            LossCallback(
                event="on_epoch_end",
                tensorboard_writer=tensorboard_writer,
                main_tag="loss/epoch"
            )
        )
    if visualize_weights:
        callbacks.add_callback(
            LayersVisualCallback(
                tensorboard_writer=tensorboard_writer,
                model=model,
                show_grad=visualize_gradients,
                use_norm=visualize_norm
            )
        )
    if visualize_waveform:
        wave_dir = None if save_dir is None else str(
            Path(save_dir, "waveform")
        )
        callbacks.add_callback(
            WaveformVisualCallback(
                tensorboard_writer=tensorboard_writer,
                model=model,
                save_dir=wave_dir,
                save_freq=save_freq
            )
        )
    if visualize_spectrogram and isinstance(model, SpectrogramMaskModel):
        spec_dir = None if save_dir is None else str(
            Path(save_dir, "spectrogram")
        )
        callbacks.add_callback(
            SpectrogramVisualCallback(
                tensorboard_writer=tensorboard_writer,
                model=model,
                save_dir=spec_dir,
                save_freq=save_freq
            )
        )
    if play_audio:
        callbacks.add_callback(
            AudioPlayerCallback(
                tensorboard_writer=tensorboard_writer,
                model=model,
                embed_residual=embed_residual,
                sample_rate=sample_rate
            )
        )
    return callbacks
