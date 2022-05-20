# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from torch.utils.tensorboard import SummaryWriter
from .visualizer import Visualizer
from .progress import ProgressBar


__all__ = ["Visualizer", "config_visualizer", "ProgressBar"]


def config_visualizer(config: dict, writer: SummaryWriter) -> Visualizer:
    """Initializes and returns a visualizer object."""
    visualizer_params = config["visualizer_params"]
    dataset_params = config["dataset_params"]
    viz = Visualizer(
        writer=writer,
        save_dir=visualizer_params["logs_path"] + "/images",
        view_spectrogram=visualizer_params["view_spectrogram"],
        view_waveform=visualizer_params["view_waveform"],
        view_gradient=visualizer_params["view_gradient"],
        play_audio=visualizer_params["play_audio"],
        num_images=visualizer_params["num_images"],
        save_image=visualizer_params["save_image"],
        save_audio=visualizer_params["save_audio"],
        save_freq=visualizer_params["save_frequency"],
        sample_rate=dataset_params["sample_rate"],
    )
    return viz
