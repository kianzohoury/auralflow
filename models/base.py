from abc import abstractmethod, ABC
from typing import List

from torchinfo import summary
from pathlib import Path

import torch
import torch.nn as nn


class SeparationModel(ABC):
    """Interface for all source separation models."""

    model: nn.Module
    optimizer: nn.Module
    batch_loss: torch.Tensor
    train_losses: List
    val_losses: List
    stop_patience: int

    def __init__(self, config: dict):
        super(SeparationModel, self).__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = config["training_params"]["checkpoint_path"]
        self.training_mode = config["training_params"]["training_mode"]
        torch.backends.cudnn.benchmark = True

    @abstractmethod
    def forward(self):
        """Forward method."""
        pass

    @abstractmethod
    def backward(self):
        """Computes batch-wise loss between estimate and target sources."""
        pass

    @abstractmethod
    def optimizer_step(self):
        """Performs gradient computation and parameter optimization."""
        pass

    @abstractmethod
    def separate(self, audio):
        pass

    def train(self):
        """Sets model to training mode."""
        self.model.train()

    def eval(self):
        """Sets model to evaluation mode."""
        self.model.eval()

    def test(self):
        """Calls forward method without gradient tracking."""
        with torch.no_grad():
            return self.forward()

    def save_model(self, global_step: int, silent=True):
        """Saves checkpoint for the model."""
        model_path = f"{self.config['model_name']}_{global_step}.pth"
        torch.save(
            self.model.cpu().state_dict(),
            Path(self.checkpoint_path) / model_path,
        )
        self.model.to(self.device)
        if not silent:
            print("Model successfully saved.")

    def load_model(self, global_step: int):
        """Loads previously trained model."""
        model_path = f"{self.config['model_name']}_{global_step}.pth"
        if Path(model_path).is_file():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Model successfully loaded.")

    def save_optim(self, global_step: int, silent=True):
        """Saves snapshot of the model's optimizer."""
        optim_path = f"{self.config['model_name']}_optim_{global_step}.pth"
        torch.save(
            self.optimizer.state_dict(),
            Path(self.checkpoint_path) / optim_path,
        )
        if not silent:
            print("Optimizer successfully saved.")

    def load_optim(self, global_step: int):
        """Loads model's optimizer to resume training."""
        optim_path = f"{self.config['model_name']}_optim_{global_step}.pth"
        if Path(optim_path).is_file():
            state_dict = torch.load(optim_path)
            self.optimizer.load_state_dict(state_dict)
            print("Optimizer successfully loaded.")

    def post_epoch_callback(self, **kwargs):
        pass

    def setup(self):
        Path(self.checkpoint_path).mkdir(exist_ok=True)

        # summary(self.model, depth=6)

    # for model in self.models:
    #     summary(model, depth=6)
