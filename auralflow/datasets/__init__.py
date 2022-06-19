# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from .datasets import AudioDataset, AudioFolder, create_audio_dataset
from .datasets import create_audio_folder, load_stems, verify_dataset
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

__all__ = [
    "AudioDataset",
    "AudioFolder",
    "create_audio_dataset",
    "create_audio_folder",
    "load_stems",
    "verify_dataset"
]

__doc__ = """Audio file I/O and dataset objects."""


def load_dataset(dataset: Dataset, training_params: dict) -> DataLoader:
    """Returns a dataloader for the dataset given some training parameters."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=training_params["batch_size"],
        num_workers=training_params["num_workers"],
        pin_memory=training_params["pin_memory"],
        persistent_workers=training_params["persistent_workers"],
        prefetch_factor=training_params["pre_fetch"],
        shuffle=True,
    )
    return dataloader
