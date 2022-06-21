# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from torch.utils.tensorboard import SummaryWriter
from .visualizer import (
    spec_show_diff,
    waveform_show_diff,
    TrainingVisualizer
)
from .progress import ProgressBar


__all__ = [
    "spec_show_diff",
    "waveform_show_diff",

    "TrainingVisualizer",
    "config_visualizer",
]





def config_visualizer(config: dict, writer: SummaryWriter) -> TrainingVisualizer:
    """Initializes and returns a visualizer object."""
    visualizer_params = config["visualizer_params"]
    dataset_params = config["dataset_params"]
    viz = TrainingVisualizer(
        writer=writer,
        output_dir=config["model_params"]["model_name"] + "/images",
        view_spectrogram=visualizer_params["view_spectrogram"],
        view_waveform=visualizer_params["view_waveform"],
        view_gradient=visualizer_params["view_gradient"],
        play_audio=visualizer_params["play_audio"],
        num_images=visualizer_params["num_images"],
        save_images=visualizer_params["save_image"],
        save_audio=visualizer_params["save_audio"],
        interval=visualizer_params["save_frequency"],
        sample_rate=dataset_params["sample_rate"],
    )
    return viz
