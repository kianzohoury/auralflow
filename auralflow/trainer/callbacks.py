# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from auralflow.models import SeparationModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .validator import run_validation_step
from auralflow.visualizer import Visualizer


__all__ = ["TrainingCallback", "WriterCallback"]


class Callback:
    def on_iteration_start(self, **kwargs):
        pass

    def on_iteration_end(self, **kwargs):
        """Post iteration callback."""
        pass

    def on_loss_end(self, **kwargs):
        """Post backward phase callback."""
        pass

    def on_epoch_start(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        """Post epoch callback."""
        pass

    def attach(self, **kwargs):
        """Attaches objects that were not passed into callback constructor."""
        _attach_to_callback(self, **kwargs)


class TrainingCallback(Callback):
    """Wrapper class for running different training callbacks."""

    model: SeparationModel
    writer: SummaryWriter
    visualizer: Visualizer
    val_dataloader: DataLoader

    def __init__(
        self,
        model: SeparationModel,
        visualizer: Visualizer,
        writer: SummaryWriter,
        val_dataloader: DataLoader,
    ) -> None:
        self.model = model
        self.visualizer = visualizer
        self.writer_ = WriterCallback(writer=writer)
        self.val_dataloader = val_dataloader

    def on_loss_end(self, global_step: int) -> None:
        # Write gradients to tensorboard.
        self.visualizer.visualize_gradient(
            model=self.model, global_step=global_step
        )

    def on_iteration_end(self, global_step: int) -> None:
        # Write iteration loss.
        self.writer_.on_iteration_end(
            model=self.model, global_step=global_step
        )

    def on_epoch_end(self, epoch: int) -> None:
        # Run validation.
        run_validation_step(
            model=self.model, val_dataloader=self.val_dataloader
        )

        # Write epoch train/val losses.
        self.writer_.write_epoch_loss(model=self.model, global_step=epoch)

        # Visualize images and audio.
        mix, target = next(iter(self.val_dataloader))
        self.visualizer.visualize(
            model=self.model, mixture=mix, target=target, global_step=epoch
        )


class WriterCallback(Callback):
    """Wrapper class for writer to handle tensorboard writing."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def update_writer(
        self, main_tag: str, named_losses: dict, global_step: int
    ) -> None:
        """Writes loss metric to tensorboard."""
        self.writer.add_scalars(main_tag, named_losses, global_step)

    def write_epoch_loss(
        self,
        model: SeparationModel,
        global_step: int,
        log_train: bool = True,
        log_val: bool = True,
        main_tag: str = "loss/epoch",
    ) -> None:
        """Writers epoch training loss to tensorboard."""
        label = f"{model.model_name}_" f"{model.training_params['criterion']}"
        named_losses = {}
        if log_train:
            train_loss_tag = f"{label}_train"
            named_losses[train_loss_tag] = model.train_losses[-1]
            self.update_writer(
                main_tag=f"{main_tag}/train",
                named_losses={train_loss_tag: model.train_losses[-1]},
                global_step=global_step,
            )
        if log_val:
            val_loss_tag = f"{label}_valid"
            named_losses[val_loss_tag] = model.val_losses[-1]
            self.update_writer(
                main_tag=f"{main_tag}/valid",
                named_losses={val_loss_tag: model.val_losses[-1]},
                global_step=global_step,
            )
        self.update_writer(
            main_tag=main_tag,
            named_losses=named_losses,
            global_step=global_step,
        )

    def on_iteration_end(
        self,
        model: SeparationModel,
        global_step: int,
        main_tag: str = "loss/iter/train",
    ) -> None:
        """Writers iteration training loss to tensorboard."""
        label = (
            f"{model.model_name}_"
            f"{model.training_params['criterion']}_train"
        )
        self.update_writer(
            main_tag=main_tag,
            named_losses={label: model.batch_loss.item()},
            global_step=global_step,
        )

    def on_epoch_end(self, model: SeparationModel, global_step: int) -> None:
        """Writers epoch train and validation loss to tensorboard."""
        self.write_epoch_loss(model=model, global_step=global_step)


def _attach_to_callback(callback: Callback, **kwargs) -> None:
    """Attaches objects to callbacks not passed in to their constructors."""
    for name, obj in kwargs:
        if hasattr(callback, name):
            setattr(callback, name, obj)
