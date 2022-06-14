# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import librosa


from auralflow.visualizer import ProgressBar
from collections import OrderedDict
from .datasets import AudioDataset, AudioFolder
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import List, Optional


__all__ = [
    "AudioDataset",
    "AudioFolder",
    "create_audio_folder",
    "audio_to_disk",
    "create_audio_dataset",
    "load_dataset",
]


def create_audio_folder(
    dataset_params: dict, subset: str = "train"
) -> datasets.AudioFolder:
    """Creates an on-the-fly streamable dataset as an AudioFolder."""
    audio_folder = AudioFolder(
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
    max_num_tracks: Optional[int] = None,
    split: str = "train",
    sample_rate: int = 44100,
    mono: bool = True,
) -> List[OrderedDict]:
    """Loads chunked audio dataset directly into disk memory."""
    audio_tracks = []
    subset_dir = list(Path(dataset_path, split).iterdir())
    max_num_tracks = max_num_tracks if max_num_tracks else float("inf")
    n_tracks = min(len(subset_dir), max_num_tracks)

    with ProgressBar(
        subset_dir, total=n_tracks, fmt=False, unit="track"
    ) as pbar:
        for index, track_folder in enumerate(pbar):
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
            duration = (
                int(librosa.get_duration(y=mixture_track, sr=44100)) * sr
            )

            entry["duration"] = duration
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
    """Helper method that creates an AudioDataset.

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
    full_dataset = audio_to_disk(
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
