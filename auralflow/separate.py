# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from argparse import ArgumentParser
from typing import Union, Mapping
from pathlib import Path

from torch import Tensor
from tqdm import tqdm

import librosa
import torch
from scipy.io import wavfile

from auralflow.build import build_model, setup_model
from auralflow.utils import load_config
from auralflow.transforms import trim_audio
from auralflow.visualizer import ProgressBar
from auralflow.models import SeparationModel


__all__ = ["separate_audio"]


def separate_audio(
    model: SeparationModel,
    filename: str,
    sr: int = 44100,
    padding: int = 200,
    duration: int = 30,
) -> Mapping[str, Tensor]:
    """Separates a single audio track and returns the stems as tensors."""

    # Get track name.
    track_path = Path(filename)
    print(track_path.name, flush=True)

    # Load audio.
    mix_audio, sr = librosa.load(
        str(track_path) + "/mixture.wav", sr=sr, duration=duration
    )
    mix_audio = torch.from_numpy(mix_audio).unsqueeze(0)

    # Split audio into chunks.
    length = model.dataset_params["sample_length"]
    step_size = length * sr
    max_frames = min(mix_audio.shape[-1], duration * sr)

    # Store chunks.
    est_chunks, res_chunks, mix_chunks = [], [], []
    offset = 0

    # Separate smaller windows of audio.
    with ProgressBar(iterable=None, total=max_frames, ascii=" #") as pbar:
        while offset < max_frames:
            # Reshape and trim audio chunk.
            audio_chunk = mix_audio[:, offset: offset + step_size - padding]

            # Unsqueeze batch dimension if not already batched.
            if audio_chunk.dim() == 2:
                audio_chunk = audio_chunk.unsqueeze(0)

            # Separate audio and trim.
            estimate = model.separate(audio_chunk)[..., :-padding]
            estimate, audio_chunk = trim_audio([estimate, audio_chunk])
            residual_chunk = audio_chunk - estimate.to(audio_chunk)

            est_chunks.append(estimate)
            res_chunks.append(residual_chunk)
            mix_chunks.append(audio_chunk)

            # Update current frame position.
            offset = offset + step_size - padding
            pbar.n += step_size - padding
            if offset + step_size >= max_frames:
                break

        pbar.n = max_frames

        # Stitch chunks to create full source estimate.
        full_estimate = torch.cat(est_chunks, dim=2).reshape(
            (mix_audio.shape[0], -1)
        )
        full_residual = torch.cat(res_chunks, dim=2).reshape(
            (mix_audio.shape[0], -1)
        )
        mix_audio = torch.cat(mix_chunks, dim=2).reshape(
            (mix_audio.shape[0], -1)
        )

        max_frames = min(
            full_estimate.shape[-1], mix_audio.shape[-1], max_frames
        )
        full_estimate = full_estimate[..., :max_frames]
        full_residual = full_residual[..., :max_frames]
        mix_audio = mix_audio[..., :max_frames]

    return {
        "estimate": full_estimate,
        "residual": full_residual,
        "mix": mix_audio,
    }


def main(
    config_filepath: str,
    audio_filepath: str,
    save_filepath: str,
    sr: int = 44100,
    padding: int = 200,
    residual: bool = True,
    duration: int = 30,
) -> None:
    """Separates audio tracks and saves them to a given filepath."""

    # Load configuration file.
    print("Reading configuration file...")
    configuration = load_config(config_filepath)
    print("  Successful.")

    # Load model. Setup restores previous state if resuming training.
    print("Loading model...")
    model = build_model(configuration)
    model = setup_model(model)
    print("  Successful.")

    # Folder to store output.
    save_dir = Path(save_filepath, "separated_audio")
    save_dir.mkdir(parents=True, exist_ok=True)

    track_paths = []
    if Path(audio_filepath).is_dir():
        for track_path in Path(audio_filepath).iterdir():
            track_paths.append(track_path)
    else:
        track_paths.append(Path(audio_filepath))

    print("Separating...")
    for track_path in track_paths:

        # Export audio.
        track_name = track_path.name
        track_dir = save_dir.joinpath(track_name)
        track_dir.mkdir(parents=True, exist_ok=True)

        # Single target for now.
        label = model.target_labels[0]

        stems = separate_audio(
            model=model,
            filename=str(track_path),
            sr=sr,
            duration=duration,
            padding=padding,
        )

        # Save outputs.
        wavfile.write(
            track_dir.joinpath(label).with_suffix(".wav"),
            rate=sr,
            data=stems["estimate"].cpu().numpy().T,
        )
        wavfile.write(
            track_dir.joinpath("mixture").with_suffix(".wav"),
            rate=sr,
            data=stems["mix"].cpu().numpy().T,
        )
        if residual:
            wavfile.write(
                track_dir.joinpath("residual").with_suffix(".wav"),
                rate=sr,
                data=stems["residual"].cpu().numpy().T,
            )
