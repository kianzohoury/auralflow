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
from typing import List, Union, Callable, Any
from auralflow.utils import load_object, save_object


class SeparationModel(ABC):
    """Interface shared among all source separation models.

    Should not be instantiated directly but rather subclassed. A
    subclass must implement the following methods: ``set_data``, ``forward``,
    ``compute_loss``, ``backward``, ``optimizer_step``, ``scheduler_step``
    and ``separate``.

    :ivar model: Underlying PyTorch model.
    :vartype model: nn.Module

    :ivar target_labels: Target source labels.
    :vartype target_labels: List[str]

    :ivar criterion: Loss function.
    :vartype criterion: Union[nn.Module, Callable]

    :ivar optimizer: Optimizer.
    :vartype optimizer: Optimizer

    :ivar scheduler: LR scheduler.
    :vartype scheduler: Any

    :ivar batch_loss: Current batch loss.
    :vartype batch_loss: FloatTensor

    :ivar train_losses: Epoch training loss history.
    :vartype train_losses: List[float]

    :ivar val_losses: Epoch validation loss history.
    :vartype val_losses: List[float]

    :ivar stop_patience: Waiting time in epochs before reducing LR, if
        validation loss does not improve.
    :vartype stop_patience: int

    :ivar max_lr_steps: Max number of LR reductions before stopping early.
    :vartype max_lr_steps: int

    :ivar use_amp: If True, uses automatic mixed precision.
    :vartype use_amp: bool

    :ivar grad_scaler: Gradient scaler (only if using automatic mixed
        precision).
    :vartype grad_scaler: Any

    :ivar is_best: Flag for checkpointing the best model.
    :vartype is_best: bool

    :ivar model_params: Model configuration parameters.
    :vartype model_params: dict

    :ivar training_params: Training configuration parameters.
    :vartype training_params: dict

    :ivar dataset_params: Dataset configuration parameters.
    :vartype dataset_params: dict

    :ivar visualizer_params: Visualizer configuration parameters.
    :vartype visualizer_params: dict

    :ivar model_name: Model name (folder name).
    :vartype model_name: dict

    :ivar checkpoint_path: Path to checkpoint folder.
    :vartype checkpoint_path: dict

    :ivar silent_checkpoint: Silences checkpointing std output.
    :vartype silent_checkpoint: dict

    :ivar training_mode: If True, model is in training mode; otherwise model
        is in inference mode.
    :vartype training_mode: dict

    :ivar device: Device.
    :vartype device: str

    Args:
        config (dict): Model configuration data.
    """

    model: nn.Module
    target_labels: List[str]
    criterion: Union[nn.Module, Callable]
    optimizer: Optimizer
    scheduler: Any
    batch_loss: FloatTensor
    train_losses: List[float]
    val_losses: List[float]
    stop_patience: int
    max_lr_steps: int
    use_amp: bool
    grad_scaler: Any
    is_best_model: int

    config: dict
    model_params: dict
    training_params: dict
    dataset_params: dict
    visualizer_params: dict
    model_name: str
    checkpoint_path: str
    silent_checkpoint: bool
    training_mode: bool
    device: str

    def __init__(self, config: dict) -> None:
        super(SeparationModel, self).__init__()

        # Store configuration settings as attributes.
        self.config = config
        self.model_params = config["model_params"]
        self.training_params = config["training_params"]
        self.dataset_params = config["dataset_params"]
        self.visualizer_params = config["visualizer_params"]
        self.model_name = self.model_params["model_name"]
        self.checkpoint_path = self.model_params["save_dir"] + "/checkpoint"
        self.silent_checkpoint = self.training_params["silent_checkpoint"]
        self.training_mode = self.training_params["training_mode"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Retrieve requested base model architecture name.
        self.base_model_type = getattr(
            importlib.import_module("auralflow.models"),
            self.model_params["model_type"],
        )

    @abstractmethod
    def set_data(self, *data: Tensor) -> None:
        """Abstract method for setting and processing data internally.

        Implementations should fill the appropriate instance attributes with
        the input data.

        Args:
            *data (Tensor): Input and/or target data.
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Abstract forward method."""
        pass

    @abstractmethod
    def compute_loss(self) -> float:
        """Abstract method for calculating the current batch-wise loss.

        Implementations should sets the ``batch_loss`` attribute and
        returns its scalar value.

        Returns:
            float: Batch-wise loss.
        """
        pass

    @abstractmethod
    def backward(self) -> None:
        """Runs gradient calculation and backpropagation."""
        pass

    @abstractmethod
    def optimizer_step(self) -> None:
        """Optimizes training loss and updates model parameters."""
        pass

    @abstractmethod
    def scheduler_step(self) -> bool:
        """Decreases the learning rate if required."""
        pass

    @abstractmethod
    def separate(self, audio: Tensor) -> Tensor:
        """Separates target sources from a mixture given its audio data.

        Args:
            audio (Tensor): Mixture audio data.

        Returns:
            Tensor: Estimated target sources.
        """
        pass

    def train(self) -> None:
        """Sets model to training mode."""
        self.model = self.model.train()

    def eval(self) -> None:
        """Sets model to evaluation mode."""
        self.model = self.model.eval()

    def test(self) -> None:
        """Calls forward method without gradient tracking."""
        self.eval()
        with torch.no_grad():
            self.forward()

    def save_model(self, global_step: int) -> None:
        """Saves the model's current state."""
        save_object(
            model=self, obj_name="model", global_step=global_step
        )

    def load_model(self, global_step: int) -> None:
        """Loads a model's previous state."""
        load_object(
            model=self, obj_name="model", global_step=global_step
        )

    def save_optim(self, global_step: int) -> None:
        """Saves the optimizer's current state."""
        save_object(
            model=self, obj_name="optimizer", global_step=global_step
        )

    def load_optim(self, global_step: int) -> None:
        """Loads an optimizer's previous state."""
        load_object(
            model=self, obj_name="optimizer", global_step=global_step
        )

    def save_scheduler(self, global_step: int) -> None:
        """Saves the scheduler's current state."""
        save_object(
            model=self, obj_name="scheduler", global_step=global_step
        )

    def load_scheduler(self, global_step: int) -> None:
        """Loads a scheduler's previous state."""
        load_object(
            model=self, obj_name="scheduler", global_step=global_step
        )

    def save_grad_scaler(self, global_step: int) -> None:
        """Saves the grad scaler's current state if using mixed precision."""
        save_object(
            model=self, obj_name="grad_scaler", global_step=global_step
        )

    def load_grad_scaler(self, global_step: int) -> None:
        """Load a grad scaler's previous state if one exists."""
        load_object(
            model=self, obj_name="grad_scaler", global_step=global_step
        )

    def save(
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
