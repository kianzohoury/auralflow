# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch.nn as nn


from abc import abstractmethod, ABC
from torch import Tensor, FloatTensor
from typing import Dict, List, Tuple, Union, OrderedDict


class SeparationModel(ABC):
    """Interface shared across all source separation models.

    Should not be instantiated directly but rather subclassed. All subclasses
    must implement the ``forward`` and ``separate`` methods.

    :ivar model: Underlying ``nn.Module`` model.
    :vartype model: nn.Module

    :ivar targets: Target source labels.
    :vartype targets: List[str]

    :ivar device: Device.
    :vartype device: str
    """

    _model: nn.Module
    _targets: List[str]
    _device: str

    def __init__(self) -> None:
        super(SeparationModel, self).__init__()

    @abstractmethod
    def forward(
        self, mixture: FloatTensor
    ) -> Union[FloatTensor, Tuple[FloatTensor, Dict[str, FloatTensor]]]:
        """Forward method.

        All subclasses must implement this method.

        Args:
            mixture (FloatTensor): Mixture data.

        Returns:
            Union[FloatTensor, Tuple[FloatTensor, Dict[str, FloatTensor]]]:
            Target source estimate or a tuple containing the target source
            estimate and a dictionary of intermediate output data.
        """
        pass

    @abstractmethod
    def separate(self, audio: Tensor) -> Tensor:
        """Separates target sources from a mixture given its audio data.

        All subclasses must implement this method.

        Args:
            audio (Tensor): Mixture audio data.

        Returns:
            Tensor: Estimated target sources.
        """
        pass

    def train(self) -> None:
        """Sets model to training mode."""
        pass
        # self._model.train()

    def eval(self) -> None:
        """Sets model to evaluation mode."""
        pass
        # self._model.eval()

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: str):
        self._model = self._model.to(device)
        self._device = device

    @property
    def targets(self):
        return self._targets

    def load_state(self, state: OrderedDict[str, Tensor], device: str) -> None:
        self._model.load_state_dict(state_dict=state)
        self.device = device


def load(model: str, target: str) -> SeparationModel:
    """Loads the weights of the pretrained, target-specific model."""
    pass
