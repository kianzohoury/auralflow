# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git
from typing import Optional

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

from auralflow.losses import get_evaluation_metrics
from auralflow.models import SeparationModel
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

    def __init__(
        self,
        model: SeparationModel,
        visualizer: Optional[Visualizer] = None,
        writer: Optional[SummaryWriter] = None,
        call_metrics: bool = True,
    ) -> None:
        self.model = model
        self.visualizer = visualizer
        self.writer = writer
        self.call_metrics = call_metrics

        if self.visualizer:
            self.visualizer_ = VisualizerCallback(
                model=model, visualizer=visualizer
            )
        if self.writer:
            self.writer_ = WriterCallback(writer=writer)
        if self.call_metrics:
            self.metrics_callback = SeparationMetricCallback(model=model)

    def on_loss_end(self, global_step: int) -> None:
        if self.visualizer:
            self.visualizer_.on_loss_end(global_step=global_step)

    def on_iteration_end(self, global_step: int) -> None:
        self.writer_.on_iteration_end(
            model=self.model, global_step=global_step
        )
        # Update global step count.
        self.model.training_params["global_step"] = global_step

    def on_epoch_end(self, mix: Tensor, target: Tensor, epoch: int) -> None:
        if self.writer:
            self.writer_.write_epoch_loss(model=self.model, global_step=epoch)
        if self.visualizer:
            self.visualizer_.on_epoch_end(mix=mix, target=target, epoch=epoch)
        if self.call_metrics:
            self.metrics_callback.on_epoch_end(mix=mix, target=target)
        # Update epoch count.
        self.model.training_params["last_epoch"] = epoch


class VisualizerCallback(Callback):
    """Callback class for training visualization tools."""

    def __init__(self, model: SeparationModel, visualizer: Visualizer):
        self.model = model
        self.visualizer = visualizer

    def on_loss_end(self, global_step: int) -> None:
        """Visualize gradients."""
        # Write gradients to tensorboard.
        self.visualizer.visualize_gradient(
            model=self.model, global_step=global_step
        )

    def on_epoch_end(self, mix: Tensor, target: Tensor, epoch: int) -> None:
        """Visualize images and play audio."""
        self.visualizer.visualize(
            model=self.model, mixture=mix, target=target, global_step=epoch
        )


class SeparationMetricCallback(Callback):
    """Callback class for printing evaluation metrics."""

    def __init__(self, model: SeparationModel, disp_freq: int = 5):
        self.model = model
        self.best_deltas = {
            "pesq": float("-inf"),
            "sar": float("-inf"),
            "sdr": float("-inf"),
            "si_sdr": float("-inf"),
            "sir": float("inf"),
            "stoi": float("-inf"),
        }
        self.disp_freq = disp_freq
        self.count = 0

    def on_epoch_end(self, mix: Tensor, target: Tensor) -> None:
        """Prints source separation evaluation metrics at epoch finish."""
        if (self.count + 1) % self.disp_freq != 0:
            self.count += 1
            return
        estimate = self.model.separate(audio=mix)
        print("Calculating evaluation metrics...")
        metrics = get_evaluation_metrics(
            mixture=mix, estimate=estimate, target=target
        )
        table = PrettyTable(["metric", "estimate", "target", "delta"])
        table.align = "l"
        estimate_metrics = list(metrics["estim_metrics"].items())
        target_metrics = list(metrics["target_metrics"].items())
        for i in range(len(estimate_metrics)):
            est_label, est_val = estimate_metrics[i]
            tar_label, tar_val = target_metrics[i]
            metric_label = "_".join(est_label.split("_")[1:])
            # Skip SIR for now.
            if metric_label == "sir":
                continue
            if est_val < float("inf") and tar_val < float("inf"):
                delta = est_val - tar_val
            else:
                delta = float("inf")
            if self.best_deltas[metric_label] < delta:
                self.best_deltas[metric_label] = delta
                delta = str(delta) + "*"
            table.add_row([metric_label, est_val, tar_val, delta])
        print(table)
        self.count = 0


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

    def write_epoch_metrics(
        self, 
        model: SeparationModel,
        global_step: int,
        log_train: bool = True,
        log_val: bool = True,
        main_tag: str = "loss/epoch",     
    ) -> None:
        

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
