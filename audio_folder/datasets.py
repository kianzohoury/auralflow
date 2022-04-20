import librosa
import numpy as np
import torch
import torchaudio
import torch.utils.data

from pathlib import Path
from torch.utils.data.dataset import IterableDataset
from typing import Iterator, List, Optional, Tuple

np.random.seed(1)


class AudioFolder(IterableDataset):
    """Ad-hoc dataset designed to generate many samples from a tiny dataset.

    Works by repeatedly resampling small chunks of full audio tracks
    with replacement. Since the pool of audio tracks to draw from is small,
    resampling many times will result in overlapping chunks, which has a
    similar effect to bootstrapping.

    Args:
        path (str): Root directory path.
        targets (List[str]): Target sources. Default: ['vocals'].
        num_samples (int): Total number of samples to generate. Default: 1e4.
        subset (str): Train or test set. Default: 'train'.
        audio_format (str): Audio format. Default: 'wav'.
        sample_rate (int): Sample rate. Default: 44100
        num_channels (int): Number of audio channels. Default: 1.
            Default: True.
        backend (str): Torchaudio backend. Default: 'soundfile'.
    """
    def __init__(self, path: str, targets: Optional[List[str]] = None,
                 sample_length: int = 3, subset: str = 'train',
                 audio_format: str = 'wav', sample_rate: int = 44100, num_channels: int = 1,
                 backend: str = 'soundfile', hop_length=None, num_fft=None, window_size=None):

        super(AudioFolder, self).__init__()
        self.path = path
        self.targets = sorted(['vocals']) if targets is None else targets
        self.sample_length = sample_length
        self.subset = subset
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.backend = backend

        self.hop_length = hop_length
        self.num_fft = num_fft
        self.window_size = window_size

        root_dir = Path(path)
        subset_dir = root_dir / subset
        track_filepaths = []
        for track_fp in subset_dir.iterdir():
            target_filepaths = [track_fp.joinpath(target).with_suffix(
                "." + audio_format) for target in self.targets]
            if all([target_fp.is_file() for target_fp in target_filepaths]):
                track_filepaths.append(track_fp)

        self._track_filepaths = track_filepaths
        # Remember the track durations to speed up the chunk sampling process.
        self._duration_cache = {}
        # Set torchaudio backend for loading (soundfile is faster for loading).
        torchaudio.set_audio_backend(backend)

    def generate_mixture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sampled_track = np.random.choice(self._track_filepaths)
        source_names = ['mixture'] + self.targets
        source_filepaths = [sampled_track.joinpath(name).with_suffix(
            "." + self.audio_format) for name in source_names]

        if sampled_track.name not in self._duration_cache:
            duration = librosa.get_duration(filename=source_filepaths[0])
            self._duration_cache[sampled_track.name] = duration
        else:
            duration = self._duration_cache[sampled_track.name]

        offset = np.random.randint(0, duration - self.sample_length)

        sources_data = []
        for source_filepath in source_filepaths:
            audio_data, _ = torchaudio.load(
                filepath=str(source_filepath),
                frame_offset=offset * self.sample_rate,
                num_frames=self.sample_length * self.sample_rate
            )
            if self.num_channels == 1:
                audio_data = torch.mean(audio_data, dim=0, keepdim=True)
            sources_data.append(audio_data)

        mix_data, target_data = sources_data[0], sources_data[1:]
        # Stack target sources along the last dimension.
        target_data = torch.stack(target_data, dim=-1)
        # Reshape mixture to match target tensor's shape.
        mix_data = mix_data.unsqueeze(-1)
        return mix_data, target_data

    def split(self, val_split: float = 0.2) -> 'audio_folder':
        """

        Args:
            val_split (float): Ratio of samples to allocate for a validation
                set. Default: 0.
        Returns:
            (audio_folder): The validation set.
        """
        assert 0 < val_split <= 1.0, \
            "Split value must be between 0.0 and 1.0."

        val_dataset = AudioFolder(
            self.path,
            self.targets,
            self.sample_length,
            self.subset,
            self.audio_format,
            self.sample_rate,
            self.num_channels,
            self.backend
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
        while True:
            mix, target = self.generate_mixture()
            yield mix, target
