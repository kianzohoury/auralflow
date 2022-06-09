---
layout: default
title: Datasets
parent: API Documentation
nav_order: 4
mathjax: true
---

# Datasets

### `AudioFolder`
```python
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
    ) -> None:
```
### `AudioDataset`
```python
class AudioDataset(Dataset):
    """Audio dataset that loads full audio tracks directly into memory."""

    def __init__(
        self,
        dataset: List,
        targets: List[str],
        chunk_size: int = 1,
        num_chunks: int = int(1e6),
        sample_rate: int = 44100,
    ) -> None:
```
### `create_audio_dataset(...)`
```python
def create_audio_dataset(
    dataset_path: str,
    targets: List[str],
    split: str = "train",
    chunk_size: int = 1,
    num_chunks: int = int(1e6),
    max_num_tracks: Optional[int] = None,
    sample_rate: int = 44100,
    mono: bool = True,
) -> AudioDataset:
    """Creates a chunked audio dataset."""
```

### Example
```python
from auralflow.datasets import create_audio_dataset


# create 100,000 3-sec chunks from a pool of 80 total tracks
train_dataset = create_audio_dataset(
    dataset_path="path/to/dataset",
    split="train",
    targets=["vocals"],
    chunk_size=3,
    num_chunks=int(1e5),
    max_num_tracks=80,
    sample_rate=44100,
    mono=True,
)

# sample mixture and target training data from the dataset
mix_audio, target_audio = next(iter(train_dataset))
```