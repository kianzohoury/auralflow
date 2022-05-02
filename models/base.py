from abc import abstractmethod, ABC
from torchinfo import summary
from pathlib import Path

import torch
import torch.nn as nn


class SeparationModel(ABC):
    """Interface for all base source separation models.

    Not meant to be implemented directly, but subclassed instead.
    All source separation models implement forward, backward, separate
    and inference methods.
    """

    def __init__(self, config: dict):
        super(SeparationModel, self).__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = config["training_params"]["checkpoint_path"]
        self.models = []
        self.losses = []
        self.optimizers = []
        self.visual_names = []
        self.is_training = config["training_params"]["training_mode"]
        torch.backends.cudnn.benchmark = True

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, **kwargs):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    @abstractmethod
    def separate(self, audio):
        pass

    @abstractmethod
    def validate(self):
        pass

    def train(self):
        """Sets each model to training mode."""
        for model in self.models:
            if isinstance(model, nn.Module):
                model.train()

    def eval(self):
        """Sets each model to evaluation mode."""
        for model in self.models:
            if isinstance(model, nn.Module):
                model.eval()

    def test(self, data):
        with torch.no_grad():
            return self.forward(data)

    def setup(self):
        for model in self.models:
            summary(model, depth=6)

    def save_checkpoint(self, global_step: int, loss: float):
        """Saves a checkpoint for each model."""
        for model in self.models:
            path = f"{model.__name__}_{global_step}_{round(loss, 5)}.pth"
            torch.save(
                model.cpu().state_dict(), Path(self.checkpoint_path) / path
            )
            model.to(self.device)


# class MaskModel(TFMaskModelBase):
#     def __init__(self, **kwargs):
#         super(MaskModel, self).__init__(**kwargs)
#
#     # def forward(self, audio) -> torch.FloatTensor:
#     #     pass
#
#
# class AudioMaskModelBase(nn.Module):
#     """Base class for deep source mask estimation directly in the time domain."""
