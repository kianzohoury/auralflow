# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch
import torch.backends.cudnn
import torch.nn as nn


from abc import abstractmethod, ABC
from torch import Tensor, FloatTensor
from torch.optim import Optimizer
from typing import Any, Callable, List, Union


class SeparationModel(ABC):
    """Interface shared across all source separation models.

    Should not be instantiated directly but rather subclassed. A
    subclass must implement the following methods: ``set_data``, ``forward``,
    ``compute_loss``, ``backward``, ``optimizer_step`` and ``separate``.

    :ivar model: Underlying ``nn.Module`` model.
    :vartype model: nn.Module

    :ivar device: Device. Uses ``'cuda'`` if available; otherwise uses
        ``'cpu'``.
    :vartype device: str

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
    """
    _model_name: str
    _checkpoint_path: str
    _silent_checkpoint: bool
    _training_mode: bool
    _stop_patience: int
    _max_lr_steps: int
    _use_amp: bool
    _grad_scaler: Any
    _is_best_model: int

    model: nn.Module
    device: str
    target_labels: List[str]
    criterion: Union[nn.Module, Callable]
    optimizer: Optimizer
    scheduler: Any
    batch_loss: FloatTensor
    train_losses: List[float]
    val_losses: List[float]

    def __init__(self) -> None:
        super(SeparationModel, self).__init__()

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
        """Abstract method computes the gradient and runs backpropagation."""
        pass

    @abstractmethod
    def optimizer_step(self) -> None:
        """Abstract method that optimizes loss and updates model parameters."""
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

    def scheduler_step(self) -> bool:
        """Decreases the learning rate if required."""
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
