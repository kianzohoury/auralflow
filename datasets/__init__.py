from torch.utils.data.dataloader import DataLoader
from . import datasets
from collections import OrderedDict
from pathlib import Path
from typing import List

import librosa
import numpy as np

from tqdm import tqdm

__all__ = ["datasets"]


def create_audio_folder(
    dataset_params: dict, subset: str = "train"
) -> datasets.AudioFolder:
    """Creates an on-the-fly streamable dataset as an AudioFolder."""
    audio_folder = datasets.AudioFolder(
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


def audio_to_disk(
    dataset_path: str, targets: List[str], split: str = "train", stop: int = 5
) -> List[OrderedDict]:
    """Loads chunked audio dataset directly into disk memory."""
    audio_tracks = []
    num_tracks = stop
    # num_tracks = len(list(Path(dataset_path, split).iterdir()))
    with tqdm(Path(dataset_path, split).iterdir(), total=num_tracks) as tq:
        entry = OrderedDict()
        for index, track_folder in enumerate(tq):
            track_name = track_folder / "mixture.wav"
            mixture_track, sr = librosa.load(track_name, sr=None)
            entry["mixture"] = mixture_track
            for target in targets:
                target_name = f"{str(track_folder)}/{target}.wav"
                entry[target], sr = librosa.load(target_name, sr=sr)
            duration = (
                int(librosa.get_duration(y=mixture_track, sr=44100)) * sr
            )
            entry["duration"] = duration
            audio_tracks.append(entry)
            if index == num_tracks:
                break
    return audio_tracks


def create_audio_dataset(
    dataset_path: str,
    targets: List[str],
    split: str = "train",
    chunk_size: int = 1,
    num_chunks: int = int(1e6),
) -> datasets.AudioDataset:
    """Creates a chunked audio dataset."""
    full_dataset = audio_to_disk(
        dataset_path=dataset_path, targets=targets, split=split
    )
    chunked_dataset = datasets.AudioDataset(
        dataset=full_dataset,
        targets=targets,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
    )
    return chunked_dataset



def load_dataset(dataset: datasets.AudioFolder, loader_params: dict) -> DataLoader:
    """Returns a dataloader for loading data from a given AudioFolder."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=loader_params["batch_size"],
        num_workers=loader_params["num_workers"],
        pin_memory=loader_params["pin_memory"],
        persistent_workers=loader_params["persistent_workers"],
    )
    return dataloader
