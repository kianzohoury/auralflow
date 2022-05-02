from torch.utils.data.dataloader import DataLoader
from .datasets import AudioFolder


def create_dataset(dataset_params: dict, subset: str = "train") -> AudioFolder:
    """Instantiates and returns an AudioFolder given a dataset config."""
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


def load_dataset(dataset: AudioFolder, loader_params: dict) -> DataLoader:
    """Returns a dataloader for loading data from a given AudioFolder."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=loader_params["batch_size"],
        num_workers=loader_params["num_workers"],
        pin_memory=loader_params["pin_memory"],
        persistent_workers=loader_params["persistent_workers"],
    )
    return dataloader
