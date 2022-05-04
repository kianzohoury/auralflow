from collections import OrderedDict
from pathlib import Path
from typing import List

import librosa
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from .datasets import AudioFolder


def create_audio_folder(
    dataset_params: dict, subset: str = "train"
) -> AudioFolder:
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


def create_audio_dataset(
    dataset_path: str, split: str, targets: List[str]
) -> List[OrderedDict]:
    """Loads full audio dataset directly into disk memory."""
    audio_tracks = []
    num_tracks = len(list(Path(dataset_path, split).iterdir()))
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
