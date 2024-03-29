# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import inspect
import os
import torch
import torch.nn as nn


from abc import ABC, abstractmethod
from auralflow.models import SeparationModel, SpectrogramNetLSTM
from auralflow.visualizer import ProgressBar
from .callbacks import CallbackManager, _create_callbacks
from copy import deepcopy
from pathlib import Path
from torch import autocast, Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List, Optional, Union, Tuple


class ModelTrainer(ABC):
    r"""Wrapper class that trains a ``SeparationModel`` model.


    All subclasses must implement the ``full_forward`` and ``scheduler_step``
    methods.

    Args:
        model (SeparationModel): Model to train.
        criterion (Union[nn.Module, Callable[..., Tensor]]): Loss criterion.
        optimizer (optional[Optimizer]): Optimizer. If ``None`` is specified,
            ``AdamW`` optimizer is used.
        scheduler (optional[object]): LR scheduler. See
            :mod:`torch.optim.lr_scheduler`. If ``None`` is specified,
            ``ReduceLROnPlateau`` is used.
        use_amp (bool): If ``True``, enables automatic mixed
            precision if possible. Default: ``True``.
        checkpoint (optional[str]): Name of the checkpoint file to save.
            If ``None`` is specified, checkpointing will be skipped.
            Default: ``checkpoint.pth``.
        logging_dir (optional[str]): Name of the directory to save tensorboard
            files to. If ``None`` is specified, tensorboard will be disabled.
            Default: ``runs``.
        device (str): Device: ``'cpu'`` | ``'cuda'``. Default: ``'cpu'``.

    Keyword Args:
        lr (Union[float, List[float, float]]]): Learning rate or a pair of
            learning rates. If two are passed in, the second value corresponds
            to the bottleneck layers. Default: ``0.008``.
        scale_grad (bool): If ``True``, gradient scaling is used if possible.
            Default: ``True``.
        clip_grad (bool): If ``True``, loss gradients are clipped. Default:
            ``True``.
        init_scale (float): Initial value for the gradient scaler, if one
            is enabled. Default: ``2.0 ** 16``.
        max_grad_norm (float): Maximum gradient norm used for
            gradient clipping, if enabled. Default: ``100.0``.
        max_plateaus (int): Maximum number of times the stop patience can
            expire before training is halted. Only applicable if the scheduler
            is an instance of
            :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.
        stop_patience (int): Number of epochs to train before stopping
            if the validation loss does not improve. Default: ``5``.
        min_delta (float): Minimum improvement in the validation loss
            required to reset the stop patience counter. Default: ``0.01``.
        view_norm (bool): If ``False``, logs the 2-norm of each
            weight/gradient if tensorboard is enabled. Default: ``True``.
        view_epoch (bool): If ``False``, logs epoch training and
            validation loss if tensorboard is enabled. Default: ``True``.
        view_iter (bool): If ``False``, logs iteration training loss if
            tensorboard is enabled. Default: ``False``.
        view_grad (bool): If ``True``, logs gradients with respect
            to layers if tensorboard is enabled. Default: ``False``.
        view_weights (bool): If ``True``, logs model weights by layer if
            tensorboard is enabled. Default: ``False``.
        view_spec (bool): If ``True``, logs target source estimates
            as spectrogram images if model is an instance of a
            ``SpectrogramMaskModel`` and tensorboard is enabled. Default:
            ``False``.
        view_wave (bool): If ``True``, logs target source estimates
            as waveform images if tensorboard is enabled. Default: ``False``.
        play_estimate (bool): If ``True``, logs listenable target source
            estimates if tensorboard is enabled. Default: ``False``.
        play_residual (bool): If ``True``, logs listenable background track
            estimates if tensorboard is enabled. Default: ``False``.
        image_dir (optional[str]):
        image_freq (int): Frequency (in epochs) at which to save images.
            Default: ``5``.
        silent (bool): If ``True``, suppresses checkpoint logging.
            Default: ``False``.
    """

    _lr: Union[float, Tuple[float, float]] = 0.008
    _scale_grad: bool = True
    _clip_grad: bool = True
    _init_scale: float = 2.0 ** 16
    _max_grad_norm: float = 100.0
    _max_plateaus: int = 5
    _stop_patience: int = 5
    _min_delta: float = 0.01
    _view_norm: bool = False
    _view_epoch: bool = False
    _view_iter: bool = False
    _view_grad: bool = False
    _view_weights: bool = False
    _view_spec: bool = False
    _view_wave: bool = False
    _play_estimate: bool = False
    _play_residual: bool = False
    _image_dir: Optional[str] = None
    _image_freq: int = 5
    _silent: bool = False

    def __init__(
        self,
        model: SeparationModel,
        criterion: Union[nn.Module, Callable[..., Tensor]],
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[object] = None,
        use_amp: bool = True,
        checkpoint: Optional[str] = "checkpoint.pth",
        logging_dir: Optional[str] = "runs",
        device: str = "cpu",
        **kwargs
    ):
        # Set device and whether to use mixed precision.
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.use_amp = use_amp and self.device == "cuda"

        # Set checkpoint and logging paths.
        self.checkpoint_path = checkpoint
        self.logging_dir = logging_dir

        self._state = {
            "last_global_step": -1,
            "last_epoch": -1,
            "stop_epoch": -1,
            "patience": self._stop_patience,
            "num_plateaus": 0,
            "train_losses": [],
            "val_losses": [],
            "best_val_loss": float("inf"),
        }

        # Set private instance attributes corresponding to specified kwargs.
        for key, val in kwargs.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", val)

        # Set model and device.
        self.model = model
        self.model.device = self.device

        # Set criterion.
        self.criterion = criterion

        # Set optimizer.
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            # Use default optimizer. Split parameters if necessary.
            if isinstance(self._lr, float):
                self._lr = (self._lr, self._lr)
            if issubclass(self.model.__class__, SpectrogramNetLSTM):
                group_1, group_2 = model.model._split_params()
                # Assign different learning rates to the groups.
                params = [
                    {"params": group_1, "lr": self._lr[0]},
                    {"params": group_2, "lr": self._lr[1]}
                ]
            else:
                params = [{"params": model.model.parameters()}]
            self.optimizer = AdamW(params=params, lr=self._lr[0])

        # Set scheduler.
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            # Use the default lr scheduler.
            self.scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                verbose=not self._silent,
                patience=self._stop_patience
            )
        
        # Enable gradient scaling and mixed precision, if possible.
        self.grad_scaler = GradScaler(
            init_scale=self._init_scale,
            enabled=self.use_amp and self._scale_grad
        )
        # Disable gradient clipping if specified.
        if not self._clip_grad:
            self._max_grad_norm = float("inf")

        # Only use a counter for default scheduler (ReduceLROnPlateau).
        if not isinstance(self.scheduler, ReduceLROnPlateau):
            self._max_plateaus = 1

    def save_state(self, checkpoint_path: str) -> None:
        if not self._silent:
            print(f"Saving checkpoint to {checkpoint_path}.")

        if not Path(checkpoint_path).exists():
            # Store private instance attributes.
            for key, val in vars(self).items():
                if key[0] == '_' and key not in {
                    "_state", "_callbacks", "_writer"
                }:
                    self._state[key[1:]] = val

        # Gather object states.
        self._state["model"] = self.model.model.cpu().state_dict()
        self._state["best_model"] = deepcopy(self._state["model"])
        self._state["optimizer"] = self.optimizer.state_dict()
        self._state["scheduler"] = self.scheduler.state_dict()
        if self.grad_scaler.is_enabled():
            self._state["grad_scaler"] = self.grad_scaler.state_dict()

        # Transfer model back to current device.
        self.model.device = self.device

        # Save checkpoint.
        torch.save(self._state, f=checkpoint_path)
        if not self._silent:
            print("  Successful")

    def load_state(self, checkpoint_path: str) -> None:
        if Path(checkpoint_path).exists():
            # Load previous trainer state.
            if not self._silent:
                print(f"Loading from {checkpoint_path}...")

            self._state = torch.load(
                f=checkpoint_path,
                map_location=self.device
            )

            # Set instance attributes.
            for key, val in self._state.items():
                if hasattr(self, f"_{key}"):
                    setattr(self, f"_{key}", val)
            
            # Load model parameters.
            self.model.load_state(
                state=self._state["model"], device=self.device
            )
            # Load optimizer.
            self.optimizer.load_state_dict(
                self._state["optimizer"]
            )
            # Load scheduler.
            self.scheduler.load_state_dict(
                self._state["scheduler"]
            )
            # Load gradient scaler.
            if self.use_amp and self._scale_grad:
                self.grad_scaler.load_state_dict(
                    self._state["grad_scaler"]
                )
            else:
                self.grad_scaler = GradScaler(enabled=False)

    def _save_if_best(self, val_loss: float):
        """Saves model only if epoch validation loss improves."""
        delta = self._state["best_val_loss"] - val_loss
        if delta >= self._min_delta:
            self._state["best_val_loss"] = val_loss
            # Reset stop patience.
            self._state["patience"] = self._stop_patience
            # Save states.
            self.save_state(checkpoint_path=self.checkpoint_path)
        else:
            self._state["patience"] -= 1
            if not self._state["patience"]:
                # Decrement total lr steps.
                self._state["num_plateaus"] += 1
                self._state["patience"] = self._stop_patience

    def _setup_callbacks(self):
        if self.logging_dir is not None:
            # Create tensorboard writer.
            if not self._silent:
                print(f"Logging tensorboard data to {self.logging_dir}.")
            self._writer = SummaryWriter(
                log_dir=self.logging_dir
            )
            # Define visualization callbacks.
            self._callbacks = _create_callbacks(
                model=self.model,
                tensorboard_writer=self._writer,
                save_dir=self._image_dir,
                save_freq=self._image_freq,
                write_iter_loss=self._view_iter,
                write_epoch_loss=self._view_epoch,
                visualize_weights=self._view_weights,
                visualize_gradients=self._view_grad,
                visualize_norm=self._view_norm,
                visualize_waveform=self._view_wave,
                visualize_spectrogram=self._view_spec,
                play_audio=self._play_estimate,
                embed_residual=self._play_residual
            )
        else:
            self._callbacks = CallbackManager()

    @abstractmethod
    def scheduler_step(self, *args, **kwargs) -> None:
        """Updates the learning rate according to the scheduler used.

        All subclasses must implement this method.

        Args:
            args: Positional arguments.

        Keyword Args:
            kwargs: Keyword arguments.
        """
        pass

    @abstractmethod
    def full_forward(self, mixture: Tensor, target: Tensor) -> Tensor:
        r"""Runs the forward pass for the model and calculates the loss.

        Gets called within the main training loop, via ``run_training(...)``
        and ``run_validation(...)``.

        All subclasses must implement this method.

        Args:
            mixture (Tensor): Mixture audio of dimension
                `(batch, channels, time)`.
            target (Tensor): Target audio of dimension `(batch, channels, time)`

        Returns:
            Tensor: The loss.

        Examples:

        >>> from auralflow.trainer import ModelTrainer
        >>> class MyCustomTrainer(ModelTrainer):
        ... # basic implementation for training spectrogram mask models.
        ...     def full_forward(self, mixture, target):
        ...         # data transformations
        ...         mix_spec, mix_phase = self.model.to_spectrogram(mixture)
        ...         target_spec, _ = self.model.to_spectrogram(target)
        ...         # model output
        ...         estimate_spec, data = self.model.forward(mix_spec)
        ...         # calculate loss
        ...         loss = self.criterion(estimate_spec, target_spec)
        ...         return loss
        """
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        resume: bool = True
    ) -> None:
        """Runs training.

        Args:
            train_loader (DataLoader): Training set dataloader.
            val_loader (DataLoader): Validation set dataloader.
            max_epochs (int): Maximum number of epochs to train for.
                Default: ``100``.
            resume (bool): If ``True``, training will be resumed from the 
                specified checkpoint file, if possible. Otherwise, training will 
                be restarted. Default: ``True``.
        """
        # Resume from previous training state.
        if resume:
            self.load_state(checkpoint_path=self.checkpoint_path)

        # Create training callbacks manager.
        self._setup_callbacks()

        if self._state["last_epoch"] >= self._state["stop_epoch"]:
            self._state["stop_epoch"] += max_epochs
        self._state["last_epoch"] += 1
        self._state["last_global_step"] += 1
        stop_epoch = self._state["stop_epoch"]
        max_iters = len(train_loader)

        for epoch in range(self._state["last_epoch"], stop_epoch):
            print(f"Epoch [{epoch}/{stop_epoch}]", flush=True)
            total_train_loss = 0

            # Set model to training mode.
            self.model.train()
            with ProgressBar(
                train_loader, total=max_iters, desc="train"
            ) as pbar:
                for idx, (mixture, target) in enumerate(pbar):
                    with autocast(
                        device_type=self.device,
                        enabled=self.use_amp,
                        dtype=torch.float16 if self.use_amp else torch.bfloat16
                    ):
                        # Run forward pass, calculate mini-batch loss.
                        batch_loss = self.full_forward(mixture, target)

                    loss_val = batch_loss.item()
                    total_train_loss += loss_val
                    mean_train_loss = total_train_loss / (idx + 1)

                    # Display loss via progress bar.
                    pbar.set_postfix({
                        "loss": f"{loss_val:6.6f}",
                        "mean_loss": f"{mean_train_loss:6.6f}",
                    })

                    # Backprop.
                    self.grad_scaler.scale(batch_loss).backward()
                    # Loss-end/post-backward phase callback.
                    self._callbacks.on_loss_end(
                        named_losses={
                            "batch_loss_train": loss_val,
                            # "total_loss_train": total_train_loss,
                        },
                        global_step=self._state["last_global_step"]
                    )
                    # Update model parameters.
                    self._optimizer_step()
                    # Iteration-end callback.
                    self._callbacks.on_iteration_end(
                        named_losses={
                            "batch_loss_train": loss_val,
                            # "total_loss_train": total_train_loss,
                        },
                        global_step=self._state["last_global_step"]
                    )
                    # Update global step count.
                    self._state["last_global_step"] += 1

            # Run validation loop. 
            mean_val_loss = self.validate(val_loader=val_loader)
            self.scheduler_step(val_loss=mean_val_loss)
            self._state["train_losses"].append(mean_train_loss)
            self._state["val_losses"].append(mean_val_loss)

            # Epoch-end callback.
            val_mixture, val_target = next(iter(val_loader))
            self._callbacks.on_epoch_end(
                named_losses={
                    "mean_loss_train": mean_train_loss,
                    "mean_loss_valid": mean_val_loss
                },
                mixture_audio=val_mixture,
                target_audio=val_target,
                epoch=epoch
            )

            # Save trainer state and objects.
            self._save_if_best(val_loss=mean_val_loss)

            if self._state["num_plateaus"] == self._max_plateaus:
                print("No improvement. Stopping training early...")
                break

            self._state["last_epoch"] += 1
            self._flush_writer()
        
        # save final state
        self.save_state(checkpoint_path=self.checkpoint_path)

    def validate(self, val_loader: DataLoader) -> float:
        """Given a validation set, runs validation and returns the mean loss.

        Args:
            val_loader (DataLoader): Validation dataloader.

        Returns:
            float: Mean validation loss.
        """
        max_iters = len(val_loader)
        total_loss = mean_loss = 0

        # Set model to validation mode.
        self.model.eval()
        with ProgressBar(
            val_loader, total=max_iters, desc="valid"
        ) as pbar:
            for idx, (mixture, target) in enumerate(pbar):
                with autocast(
                    device_type=self.device,
                    enabled=self.use_amp,
                    dtype=torch.float16 if self.use_amp else torch.bfloat16
                ):
                    with torch.no_grad():
                        # Run forward pass, calculate mini-batch loss.
                        batch_loss = self.full_forward(mixture, target)
                        loss_val = batch_loss.item()
                        total_loss += loss_val
                        mean_loss = total_loss / (idx + 1)

                # Display loss.
                pbar.set_postfix({
                    "loss": f"{loss_val:6.6f}",
                    "mean_loss": f"{mean_loss:6.6f}",
                })
        return mean_loss

    def _optimizer_step(self) -> None:
        """Optimizes and updates the model's parameters."""
        self.grad_scaler.unscale_(self.optimizer)

        # Clip gradient.
        _ = nn.utils.clip_grad_norm_(
            self.model.model.parameters(), max_norm=self._max_grad_norm
        )
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # Quicker gradient zeroing.
        for param in self.model.model.parameters():
            param.grad = None

    def _flush_writer(self):
        if self.logging_dir is not None:
            self._writer.flush()

    # @property
    # def scheduler(self):
    #     return self._scheduler

    # @scheduler.setter
    # def scheduler(self, scheduler_obj: object) -> None:
    #     if hasattr(scheduler_obj, "state_dict"):
    #         is_valid = callable(scheduler_obj.state_dict)
    #     else:
    #         raise AttributeError("Scheduler must define state_dict().")
    #     if hasattr(scheduler_obj, "load_state_dict"):
    #         is_valid = is_valid and callable(scheduler_obj.load_state_dict)
    #     else:
    #         raise AttributeError("Scheduler must define load_state_dict().")
    #     if not is_valid:
    #         raise ValueError(
    #             f"{scheduler_obj.__class__.__name__} is not a valid scheduler."
    #         )
    #     else:
    #         self._scheduler = scheduler_obj

# import time
# def time_forward(full_forward):
#     def inner(*args, **kwargs):
#         t1 = time.clock()
#         result = full_forward(*args, **kwargs)
#         t2 = time.clock()
#         print(f"TIME: {t2 - t1}")
#         return result
#     return inner
        

class _DefaultModelTrainer(ModelTrainer):
    """Default ``ModelTrainer`` class."""

    scheduler: ReduceLROnPlateau

    def __init__(self, *args, **kwargs) -> None:
        super(_DefaultModelTrainer, self).__init__(*args, **kwargs)

    # @time_forward
    def full_forward(self, mixture: Tensor, target: Tensor) -> Tensor:
        with autocast(
            device_type=self.device,
            enabled=self.use_amp,
            dtype=torch.float16 if self.use_amp else torch.bfloat16
        ):
            if hasattr(self.criterion, "_forward_wrapper"):
                loss = self.criterion._forward_wrapper(
                        model=self.model, mix_audio=mixture, target_audio=target
                )
                return loss
            else:
                raise AttributeError(
                    "Criterion must define the _forward_wrapper() method."
                )

    def scheduler_step(self, val_loss: float) -> None:
        self.scheduler.step(val_loss)


    
