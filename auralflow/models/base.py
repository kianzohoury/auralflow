# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import importlib
import torch
import torch.backends.cudnn
import torch.nn as nn


from abc import abstractmethod, ABC
from torch import Tensor, FloatTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Union, Callable, Any
from auralflow.utils import load_object, save_object


class SeparationModel(ABC):
    """Interface shared among all source separation models."""

    model: nn.Module
    target_labels: List[str]
    criterion: Union[nn.Module, Callable]
    optimizer: Optimizer
    scheduler: ReduceLROnPlateau
    batch_loss: FloatTensor
    train_losses: List[float]
    val_losses: List[float]
    stop_patience: int
    max_lr_steps: int
    grad_scaler: Any
    use_amp: bool
    is_best_model: bool

    def __init__(self, config: dict):
        super(SeparationModel, self).__init__()

        # Store configuration settings as attributes.
        self.config = config
        self.model_params = config["model_params"]
        self.training_params = config["training_params"]
        self.dataset_params = config["dataset_params"]
        self.visualizer_params = config["visualizer_params"]
        self.checkpoint_path = self.training_params["checkpoint_path"]
        self.silent_checkpoint = self.training_params["silent_checkpoint"]
        self.model_name = self.model_params["model_name"]
        self.training_mode = self.training_params["training_mode"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Retrieve requested base model architecture name.
        self.base_model_type = getattr(
            importlib.import_module("models"), self.model_params["model_type"]
        )

    @abstractmethod
    def set_data(self, *args) -> None:
        """Set and process data for internal access."""
        pass

    @abstractmethod
    def forward(self) -> None:
        """Forward method."""
        pass

    @abstractmethod
    def compute_loss(self) -> float:
        """Updates and returns the current batch-wise loss."""
        pass

    @abstractmethod
    def backward(self) -> None:
        """Computes batch-wise loss between estimate and target sources."""
        pass

    @abstractmethod
    def optimizer_step(self) -> None:
        """Performs gradient computation and parameter optimization."""
        pass

    @abstractmethod
    def scheduler_step(self) -> bool:
        """Decreases learning rate if necessary."""
        pass

    @abstractmethod
    def separate(self, audio: Tensor) -> Tensor:
        """Separates target source from mixture audio."""
        pass

    def train(self) -> None:
        """Sets model to training mode."""
        self.model = self.model.train()

    def eval(self) -> None:
        """Sets model to evaluation mode."""
        self.model = self.model.eval()

    def test(self):
        """Calls forward method without gradient tracking."""
        self.eval()
        with torch.no_grad():
            return self.forward()

    def save_model(self, global_step: int) -> None:
        """Saves the model's current state."""
        save_object(
            model_wrapper=self, obj_name="model", global_step=global_step
        )

    def load_model(self, global_step: int) -> None:
        """Loads a model's previous state."""
        load_object(
            model_wrapper=self, obj_name="model", global_step=global_step
        )

    def save_optim(self, global_step: int) -> None:
        """Saves the optimizer's current state."""
        save_object(
            model_wrapper=self, obj_name="optimizer", global_step=global_step
        )

    def load_optim(self, global_step: int) -> None:
        """Loads an optimizer's previous state."""
        load_object(
            model_wrapper=self, obj_name="optimizer", global_step=global_step
        )

    def save_scheduler(self, global_step: int) -> None:
        """Saves the scheduler's current state."""
        save_object(
            model_wrapper=self, obj_name="scheduler", global_step=global_step
        )

    def load_scheduler(self, global_step: int) -> None:
        """Loads a scheduler's previous state."""
        load_object(
            model_wrapper=self, obj_name="scheduler", global_step=global_step
        )

    def save_grad_scaler(self, global_step: int) -> None:
        """Saves the grad scaler's current state if using mixed precision."""
        save_object(
            model_wrapper=self, obj_name="grad_scaler", global_step=global_step
        )

    def load_grad_scaler(self, global_step: int) -> None:
        """Load a grad scaler's previous state if one exists."""
        load_object(
            model_wrapper=self, obj_name="grad_scaler", global_step=global_step
        )

    def save_all(
        self,
        global_step: int,
        model: bool = True,
        optim: bool = True,
        scheduler: bool = True,
        grad_scaler: bool = True,
    ) -> None:
        """Saves all training objects in one call."""
        if model:
            self.save_model(global_step=global_step)
        if optim:
            self.save_optim(global_step=global_step)
        if scheduler:
            self.save_scheduler(global_step=global_step)
        if grad_scaler:
            self.save_grad_scaler(global_step=global_step)

    def pre_epoch_callback(self, *args, **kwargs):
        pass

    def mid_epoch_callback(self, *args, **kwargs):
        pass

    def post_epoch_callback(self, *args, **kwargs):
        pass
