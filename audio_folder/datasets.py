import librosa
import numpy as np
import torch
import torchaudio
import torch.utils.data

from pathlib import Path
from torch.utils.data.dataset import IterableDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Iterator, List, Optional, Tuple
from tqdm import tqdm


class AudioFolder(IterableDataset):
    """An on-the-fly audio sample generator designed to be memory efficient.

    Similar to PyTorch's ImageFolder class, it loads audio clips from a
    an audio folder with a specific file structure. Loading audio files
    especially uncompressed formats (e.g. .wav), tend to increase memory usage
    and slow down runtime if utilizing GPUs.

    * Instead of chunking each track and loading an entire audio folder's worth
      of chunks, samples are randomly (with replacement) as needed by the
      dataloader. Note that when a dataloader has multiple workers and memory
      is pinned, both the sampling process and data transfer to GPU are sped up
      considerably, making on-the-fly data generation a viable option.

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

    Examples:
        >>> train_data = AudioFolder('toy_dataset/wav', ['vocals'],
        ... subset='train')
        >>> audio_sample = next(iter(train_data))
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
        # Remember the track durations to speed up the chunk sampling process.
        self._duration_cache = {}
        # Set torchaudio backend for loading (soundfile > sox_io for loading).
        torchaudio.set_audio_backend(backend)
        np.random.seed(1)

    def _generate_mixture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates audio mixtures and their ground-truth target sources.

        Returns:
            (tuple): A tuple of a training sample and its target sources.

        Raises:
            ValueError: If the audio backend cannot read an audio format.
        """
        sampled_track = np.random.choice(self._track_filepaths)
        source_names = ["mixture"] + self.targets
        source_filepaths = [
            sampled_track.joinpath(name).with_suffix("." + self.audio_format)
            for name in source_names
        ]
        if sampled_track.name not in self._duration_cache:
            duration = librosa.get_duration(filename=source_filepaths[0])
            self._duration_cache[sampled_track.name] = duration
        else:
            duration = self._duration_cache[sampled_track.name]
        offset = np.random.randint(0, duration - self.sample_length)

        sources_data = []
        for source_filepath in source_filepaths:
            try:
                audio_data, _ = torchaudio.load(
                    filepath=str(source_filepath),
                    frame_offset=offset * self.sample_rate,
                    num_frames=self.sample_length * self.sample_rate,
                )
            except ValueError as e:
                raise ValueError(f"Cannot load {str(source_filepath)}.") from e
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
            >>> train_data = AudioFolder('toy_dataset/wav', ['vocals'],
            ... subset='train')
            >>> val_data = train_data.split(val_split=0.2)
        """
        assert 0 < val_split <= 1.0, "Split value must be between 0.0 and 1.0."

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
        """Iter method called by the dataloader."""
        while True:
            mix, target = self._generate_mixture()
            yield mix, target



class AudioDataset(Dataset):
    def __init__(self, audio_folder, num_samples=1000):
        self.dataset = []
        self.num_samples = num_samples
        dataloader = DataLoader(audio_folder, num_workers=16, pin_memory=True)
        with tqdm(dataloader, total=num_samples) as tq:
            for i, sample in enumerate(tq):
                self.dataset.append(sample)
                if i == num_samples:
                    break
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.num_samples

class StreamDataset(Dataset):
    def __init__(self, audio_folder):
        frame_length = (2048 * 44100) // 22050
        hop_length = (512 * 44100) // 22050
        self.data = []
        total = 0
        for filepath in audio_folder._track_filepaths:

            # Stream the data, working on 128 frames at a time
            stream = librosa.stream(filepath / "mixture.wav",
                                    block_length=128,
                                    frame_length=frame_length,
                                    hop_length=hop_length)

            chromas = []
            for y in stream:
                chroma_block = librosa.feature.chroma_stft(y=y, sr=44100,
                                                            n_fft=frame_length,
                                                            hop_length=hop_length,
                                                            center=True)
                total += 1
            chromas.append(chromas)
            self.data.extend(chromas)
            print(len(self.data))


