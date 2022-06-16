# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import librosa
import numpy as np

from auralflow.visualizer import ProgressBar
from collections import OrderedDict
from .datasets import AudioDataset, AudioFolder
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import List, Mapping, Optional

__all__ = [
    "AudioDataset",
    "AudioFolder",
    "create_audio_dataset",
    "create_audio_folder",
    "load_audio",
]


def create_audio_folder(
    dataset_path: str,
    targets: List[str],
    split: str = "train",
    chunk_size: int = 2,
    sample_rate: int = 44100,
    mono: bool = True,
    audio_format: str = "wav",
    backend: str = "soundfile",
) -> AudioFolder:
    """Helper method that creates an ``AudioFolder``.

    Args:
        dataset_path (str): Path to the directory of audio files belonging to
            a dataset.
        targets (List[str]): Labels of the ground-truth source signals.
        split (str): Subset of the directory to read from. Default: "train".
        chunk_size (int): Duration of each resampled audio chunk in seconds.
            Default: 2.
        sample_rate (int): Sample rate. Default: 44100.
        mono (bool): Load tracks as mono. Default: True.
        audio_format (str): Audio format. Default: 'wav'.
        backend (str): Torchaudio backend. Default: 'soundfile'.

    Returns:
        AudioFolder: Audio folder.
    """
    audio_folder = AudioFolder(
        dataset_path=dataset_path,
        targets=targets,
        subset=split,
        sample_length=chunk_size,
        sample_rate=sample_rate,
        num_channels=1 if mono else 2,
        audio_format=audio_format,
        backend=backend
    )
    return audio_folder


def load_audio(
    dataset_path: str,
    targets: List[str],
    split: str = "train",
    max_num_tracks: Optional[int] = None,
    sample_rate: int = 44100,
    mono: bool = True,
) -> List[OrderedDict[..., Mapping[str, np.ndarray], Mapping[str, int]]]:
    """Loads audio data directly into memory.

    Args:
        dataset_path (str): Path to the directory of audio files belonging to
            a dataset.
        targets (List[str]): Labels of the ground-truth source signals.
        split (str): Subset of the directory to read from. Default: "train".
        max_num_tracks (int): Max number of full audio tracks to resample from.
            Default: 100.
        sample_rate (int): Sample rate. Default: 44100.
        mono (bool): Load tracks as mono. Default: True.

    Returns:
        List[..., OrderedDict[Mapping[str, np.ndarray], Mapping[str, int]]]:
            Audio data corresponding to mixture, (1 or more targets), duration.
    """
    audio_tracks = []
    subset_dir = list(Path(dataset_path, split).iterdir())
    max_num_tracks = max_num_tracks if max_num_tracks else float("inf")
    n_tracks = min(len(subset_dir), max_num_tracks)

    with ProgressBar(
        subset_dir, total=n_tracks, fmt=False, unit="track"
    ) as pbar:
        for index, track_folder in enumerate(pbar):

            # Create entry.
            entry = OrderedDict()
            track_name = track_folder / "mixture.wav"
            if not track_name.is_file():
                continue

            # Load mixture and target tracks.
            mixture_track, sr = librosa.load(track_name, sr=sample_rate)
            entry["mixture"] = mixture_track
            for target in sorted(targets):
                target_name = f"{str(track_folder)}/{target}.wav"
                entry[target], sr = librosa.load(target_name, sr=sr, mono=mono)

            # Record duration of mixture track.
            num_seconds = librosa.get_duration(y=mixture_track, sr=44100)
            duration = int(num_seconds * sr)
            entry["duration"] = duration

            # Store entry.
            audio_tracks.append(entry)
            if index == n_tracks:
                break
    return audio_tracks


def create_audio_dataset(
    dataset_path: str,
    targets: List[str],
    split: str = "train",
    chunk_size: int = 2,
    num_chunks: int = 10000,
    max_num_tracks: int = 100,
    sample_rate: int = 44100,
    mono: bool = True,
) -> AudioDataset:
    """Helper method that creates an ``AudioDataset``.

    Args:
        dataset_path (str): Path to the directory of audio files belonging to
            a dataset.
        targets (List[str]): Labels of the ground-truth source signals.
        split (str): Subset of the directory to read from. Default: "train".
        chunk_size (int): Duration of each resampled audio chunk in seconds.
            Default: 2.
        num_chunks (int): Number of resampled chunks to create. Default: 10000.
        max_num_tracks (int): Max number of full audio tracks to resample from.
            Default: 100.
        sample_rate (int): Sample rate. Default: 44100.
        mono (bool): Load tracks as mono. Default: True.

    Returns:
        AudioDataset: Audio dataset.
    """
    # Full-length audio tracks.
    full_dataset = load_audio(
        dataset_path=dataset_path,
        targets=targets,
        split=split,
        max_num_tracks=max_num_tracks,
        sample_rate=sample_rate,
        mono=mono,
    )
    # Chunked dataset.
    chunked_dataset = AudioDataset(
        dataset=full_dataset,
        targets=targets,
        chunk_size=chunk_size,
        num_chunks=int(num_chunks),
        sample_rate=sample_rate,
    )
    return chunked_dataset


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
