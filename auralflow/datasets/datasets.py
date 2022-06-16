# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import librosa
import numpy as np
import torch
import torchaudio


from auralflow.visualizer import ProgressBar
from pathlib import Path
from torch import Tensor
from torch.utils.data.dataset import Dataset, IterableDataset
from typing import Iterator, List, Optional, Tuple


class AudioFolder(IterableDataset):
    """An on-the-fly audio sample generator designed to be memory efficient.

    Similar to PyTorch's ImageFolder class, it loads audio from an audio
    folder without storing the audio directly in memory. Instead, chunks of
    audio are randomly sampled (with replacement) by the dataloader.

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

    def _generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method that generates pairs of mixture-target audio data."""
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
        """Splits itself into training and validation sets.

        Args:
            val_split (float): Ratio of training files to split off into
                the validation ``AudioFolder``. Default 0.2.

        Returns:
            AudioFolder: The validation ``AudioFolder``.
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

    def __iter__(self) -> Tuple[Tensor, Tensor]:
        """Yields pairs of audio data iteratively.

        Yields:
            Tuple[Tensor, Tensor]: Mixture and target data, respectively.
        """
        while True:
            mix, target = self._generate_sample()
            yield mix, target

    def __getitem__(self, idx: int) -> str:
        """Returns the name of the track at the index in the folder."""
        return str(self._track_filepaths[idx])


class AudioDataset(Dataset):
    """Audio dataset that loads full audio tracks directly into memory.

    Args:
        dataset (List): Dataset data.
        targets (List[str]): Labels of the ground-truth source signals.
        chunk_size (int): Duration of each resampled audio chunk in seconds.
            Default: 2.
        num_chunks (int): Number of resampled chunks to create. Default: 10000.
    """

    def __init__(
        self,
        dataset: List,
        targets: List[str],
        chunk_size: int = 3,
        num_chunks: int = int(1e4),
        sample_rate: int = 44100,
    ):
        super(AudioDataset, self).__init__()
        self.targets = targets
        self.dataset = _make_chunks(
            dataset=dataset,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            sr=sample_rate,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Gets the audio data at the corresponding index.

        Returns:
            Tuple[Tensor, Tensor]: Mixture and target data, respectively.
        """
        mixture, targets = self.dataset[idx]
        return mixture, targets


def _make_chunks(
    dataset: List,
    chunk_size: int,
    num_chunks: int,
    sr: int = 44100,
    energy_cutoff: float = 0.1,
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

                # Un-squeeze channels dimension if audio is mono.
                if mix_chunk.dim() == 1:
                    mix_chunk = mix_chunk.unsqueeze(0)
                    target_chunks = target_chunks.unsqueeze(0)
                chunked_dataset.append([mix_chunk, target_chunks])

            if index == num_chunks:
                break
    del dataset
    return chunked_dataset
