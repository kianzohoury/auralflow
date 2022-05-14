import os
from collections import OrderedDict
from pathlib import Path
from typing import List
from typing import Optional

import io
import librosa
import requests
import zipfile
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from visualizer.progress import ProgressBar
from . import datasets

# import drive

ID = "1mbIa4kJWaYfaXr54EMwLaq9XNtNesxUE"


def create_audio_folder(
    dataset_params: dict, subset: str = "train"
) -> datasets.AudioFolder:
    """Creates an on-the-fly streamable dataset as an AudioFolder."""
    audio_folder = datasets.AudioFolder(
        subset=subset,
        dataset_path=dataset_params["dataset_path"],
        targets=dataset_params["target"],
        sample_length=dataset_params["sample_length"],
        audio_format=dataset_params["audio_format"],
        sample_rate=dataset_params["sample_rate"],
        num_channels=dataset_params["num_channels"],
        backend=dataset_params["backend"],
    )
    return audio_folder


def audio_to_disk(
    dataset_path: str,
    targets: List[str],
    max_num_tracks: int,
    split: str = "train"
) -> List[OrderedDict]:
    """Loads chunked audio dataset directly into disk memory."""
    audio_tracks = []
    subset_dir = list(Path(dataset_path, split).iterdir())
    num_tracks = min(len(subset_dir), max_num_tracks)

    with ProgressBar(
        subset_dir, total=num_tracks, fmt=False, unit="track"
    ) as tq:

        for index, track_folder in enumerate(tq):
            entry = OrderedDict()
            track_name = track_folder / "mixture.wav"
            if not track_name.is_file():
                continue

            mixture_track, sr = librosa.load(track_name, sr=None)
            entry["mixture"] = mixture_track

            for target in sorted(targets):
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
    normalize: bool = False,
    max_num_tracks: Optional[int] = None
) -> datasets.AudioDataset:
    """Creates a chunked audio dataset."""
    full_dataset = audio_to_disk(
        dataset_path=dataset_path, targets=targets, split=split, max_num_tracks=max_num_tracks
    )
    chunked_dataset = datasets.AudioDataset(
        dataset=full_dataset,
        targets=targets,
        chunk_size=chunk_size,
        num_chunks=int(num_chunks),
    )
        
    return chunked_dataset


def load_dataset(dataset: Dataset, training_params: dict) -> DataLoader:
    """Returns a dataloader for a given dataset."""
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


def download_musdb18() -> None:
    """Downloads waveform musdb18 dataset."""
    dataset_dir = Path(os.getcwd(), "wav")

    dataset_dir.mkdir(exist_ok=True)
    request = requests.get(
        f"https://drive.google.com/uc?export=download&confirm=9_s_&id={ID}"
    )
    with zipfile.ZipFile(io.BytesIO(request.content)) as zip_file:
        pass
        # for track_file in tqdm(zip_file.infolist(), desc="Downloading..."):
        #     try:
        #         zip_file.extract(track_file, path=dataset_dir)
        #     except zipfile.error as e:
        #         pass
