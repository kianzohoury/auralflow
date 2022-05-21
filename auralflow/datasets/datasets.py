# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import librosa
import numpy as np
import torch
import torchaudio


from pathlib import Path
from torch import Tensor
from torch.utils.data.dataset import IterableDataset, Dataset
from typing import Iterator, List, Optional, Tuple
from auralflow.visualizer import ProgressBar


class AudioFolder(IterableDataset):
    """An on-the-fly audio sample generator designed to be memory efficient.

    Similar to PyTorch's ImageFolder class, it loads audio clips from a
    an audio folder with a specific file structure. Loading audio files
    especially uncompressed formats (e.g. .wav), tend to increase memory usage
    and slow down runtime if utilizing GPUs.

    * Instead of chunking each track and loading an entire audio folder's worth
      of chunks, samples are randomly (with replacement) as needed by the
      dataloader. Note that when a dataloader has multiple workers and memory
      is pinned, both the sampling process and audio transfer to GPU are sped
      up considerably, making on-the-fly audio generation a viable option.

    * If an audio folder consists of just a few tracks, resampling can generate
      a much larger dataset via chunking. However, resampling will eventually
      result in overlapping chunks, which may reduce sample variance due to
      the same effect that bootstrapping creates.

    Args:
        dataset_path (str): Root directory path.
        targets (List[str]): Target sources. Default: ['vocals'].
        sample_length (int): The duration of an audio sample.
        subset (str): Train or test set. Default: 'train'.
        audio_format (str): Audio format. Default: 'wav'.
        sample_rate (int): Sample rate. Default: 44100
        num_channels (int): Number of audio channels. Default: 1.
            Default: True.
        backend (str): Torchaudio backend. Default: 'soundfile'.
    """

    def __init__(
        self,
        dataset_path: str,
        targets: Optional[List[str]] = None,
        sample_length: int = 3,
        subset: str = "train",
        audio_format: str = "wav",
        sample_rate: int = 44100,
        num_channels: int = 1,
        backend: str = "soundfile",
    ):
        super(AudioFolder, self).__init__()
        self.path = dataset_path
        self.targets = ["vocals"] if targets is None else sorted(targets)
        self.sample_length = sample_length
        self.subset = subset
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.backend = backend

        root_dir = Path(dataset_path)
        subset_dir = root_dir / subset
        track_filepaths = []
        for track_fp in subset_dir.iterdir():
            target_filepaths = [
                track_fp.joinpath(target).with_suffix("." + audio_format)
                for target in self.targets
            ]
            if all([target_fp.is_file() for target_fp in target_filepaths]):
                track_filepaths.append(track_fp)

        self._track_filepaths = track_filepaths
        # Cache the track durations to speed up sampling.
        self._duration_cache = {}

        # Set torchaudio backend for audio loading.
        torchaudio.set_audio_backend(backend)
        np.random.seed(1)

    def _generate_mixture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates audio mixture and their ground-truth target sources.

        Returns:
            (tuple): A tuple of a training sample and its target sources.

        Raises:
            ValueError: If the audio backend cannot read an audio format.
        """
        sampled_track = np.random.choice(self._track_filepaths)
        source_names = ["mixture"] + self.targets
        src_filepaths = []

        for name in source_names:
            src_filepaths.append(
                sampled_track.joinpath(name).with_suffix(
                    "." + self.audio_format
                )
            )

        if sampled_track.name not in self._duration_cache:
            duration = librosa.get_duration(filename=src_filepaths[0])
            self._duration_cache[sampled_track.name] = duration
        else:
            duration = self._duration_cache[sampled_track.name]
        offset = np.random.randint(0, duration - self.sample_length)

        sources_data = []
        for source_filepath in src_filepaths:
            try:
                audio_data, _ = torchaudio.load(
                    filepath=str(source_filepath),
                    frame_offset=offset * self.sample_rate,
                    num_frames=self.sample_length * self.sample_rate,
                )
            except IOError as e:
                raise IOError(f"Cannot load {str(source_filepath)}.") from e
            if self.num_channels == 1:
                audio_data = torch.mean(audio_data, dim=0, keepdim=True)
            sources_data.append(audio_data)

        mix_data, target_data = sources_data[0], sources_data[1:]
        # Stack target sources along the last dimension.
        target_data = torch.stack(target_data, dim=-1)

        # Reshape mixture to match target tensor's shape.
        mix_data = mix_data.unsqueeze(-1)
        return mix_data, target_data

    def split(self, val_split: float = 0.2) -> "AudioFolder":
        """Splits the audio folder into training and validation folders.

        Args:
            val_split (float): Ratio of samples to allocate for a validation
                set. Default: 0.
        Returns:
            (AudioFolder): The validation set.

        Example:
        """
        val_split = 0.2 if (val_split > 1 or val_split < 0) else val_split

        val_dataset = AudioFolder(
            self.path,
            self.targets,
            self.sample_length,
            self.subset,
            self.audio_format,
            self.sample_rate,
            self.num_channels,
            self.backend,
        )

        # Shuffle and get the split point.
        np.random.shuffle(self._track_filepaths)
        split_index = round(len(self._track_filepaths) * (1.0 - val_split))

        # Make the split & update the pointers.
        train_filepaths = self._track_filepaths[:split_index]
        val_filepaths = self._track_filepaths[split_index:]
        self._track_filepaths = train_filepaths
        val_dataset._track_filepaths = val_filepaths
        return val_dataset

    def __iter__(self) -> Iterator:
        """Iter method."""
        while True:
            mix, target = self._generate_mixture()
            yield mix, target


class AudioDataset(Dataset):
    """Audio dataset that loads full audio tracks directly into memory."""

    def __init__(
        self,
        dataset: List,
        targets: List[str],
        chunk_size: int = 1,
        num_chunks: int = int(1e6),
        sample_rate: int = 44100,
    ):
        super(AudioDataset, self).__init__()
        self.targets = targets
        self.dataset = make_chunks(
            dataset=dataset,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            sr=sample_rate,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        mixture, targets = self.dataset[idx]
        return mixture, targets


def make_chunks(
    dataset: List,
    chunk_size: int,
    num_chunks: int,
    sr: int = 44100,
    energy_cutoff: float = 1.0,
) -> List[List[Tensor]]:
    """Transforms an audio dataset into a chunked dataset."""
    chunked_dataset = []
    num_tracks = len(dataset)

    with ProgressBar(
        range(num_chunks), total=num_chunks, fmt=False, unit="chunk"
    ) as pbar:
        for index, _ in enumerate(pbar):
            discard_entry = False
            entry = dataset[np.random.randint(num_tracks)]
            mixture = entry["mixture"]
            duration = entry["duration"]
            offset = np.random.randint(0, duration - chunk_size * sr)
            stop = offset + int(sr * chunk_size)
            mix_chunk = torch.from_numpy(mixture[offset:stop])

            # Discard silent entries.
            if torch.linalg.norm(mix_chunk) < energy_cutoff:
                continue

            target_tensors = []
            for target_name, target_data in list(entry.items())[1:-1]:
                target_chunk = torch.from_numpy(target_data[offset:stop])

                # Discard silent entries.
                if torch.linalg.norm(target_chunk) < energy_cutoff:
                    discard_entry = True
                    break
                else:
                    target_tensors.append(target_chunk)

            if not discard_entry:
                target_chunks = torch.stack(target_tensors, dim=-1)

                # Unsqueeze channels dim if audio is mono.
                if mix_chunk.dim() == 1:
                    mix_chunk = mix_chunk.unsqueeze(0)
                    target_chunks = target_chunks.unsqueeze(0)
                chunked_dataset.append([mix_chunk, target_chunks])

            if index == num_chunks:
                break
    del dataset
    return chunked_dataset


# def normalize_dataset(dataset, ratio: float = 0.2):
#     sr = 44100
#     chunk_size = dataset[0]["mixture"].shape[0] // sr
#     mix_sum, mix_sum_square = torch.zeros((sr * chunk_size)), torch.zeros(
#         (sr * chunk_size)
#     )
#     with tqdm(iter(dataset), total=int(len(dataset) * ratio)) as tq:
#         for index, track in enumerate(tq):
#             mixture = track["mixture"]
#
#             mix_sum += mixture
#             mix_sum_square += mixture**2
#             if index == int(len(dataset) * ratio):
#                 break
#
#     mix_mean = mix_sum / (int(len(dataset) * ratio))
#     mix_std = torch.sqrt(
#         mix_sum_square / (int(len(dataset) * ratio)) - mix_mean * mix_mean
#     )
#     with tqdm(iter(dataset), total=len(dataset)) as tq:
#         for index, track in enumerate(tq):
#             track["mixture"] = (track["mixture"] - mix_mean) / (mix_std + 1e-9)
#             if index == len(dataset):
#                 break
#     print(f"Dataset statistics: mean: {mix_mean}, std: {mix_std}")
