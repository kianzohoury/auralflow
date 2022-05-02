import inspect
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
from .datasets import AudioFolder, AudioDataset, StreamDataset


def create_dataset(dataset_params: dict, subset: str = "train") -> AudioFolder:
    """Instantiates and returns an AudioFolder.

    Args:
        dataset_params (dict): Dataset parameters.
        subset (int): Subset of the dataset. Default: "train".
    """
    audio_folder = AudioFolder(
        subset=subset,
        dataset_path=dataset_params["dataset_path"],
        targets=dataset_params["targets"],
        sample_length=dataset_params["sample_length"],
        audio_format=dataset_params["audio_format"],
        sample_rate=dataset_params["sample_rate"],
        num_channels=dataset_params["num_channels"],
        backend=dataset_params["backend"],
    )
    return audio_folder


def load_dataset(dataset: AudioFolder, dataset_params: dict) -> DataLoader:
    """Instantiates a dataloader."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataset_params["loader_params"]["batch_size"],
        num_workers=dataset_params["loader_params"]["num_workers"],
        pin_memory=dataset_params["loader_params"]["pin_memory"],
        persistent_workers=dataset_params["loader_params"]["persistent_workers"],
        prefetch_factor=4
    )
    return dataloader
