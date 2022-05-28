# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

import librosa
import torch
from scipy.io import wavfile

from auralflow.models import create_model, setup_model
from auralflow.utils import load_config


def main(
    config_filepath: str,
    audio_filepath: str,
    save_filepath: str,
    sr: int = 44100,
    padding: int = 200,
    residual: bool = True,
    duration: int = 30
) -> None:
    """Separates a full audio track and saves it."""

    # Load configuration file.
    print("-" * 79 + "\nReading configuration file...")
    configuration = load_config(config_filepath)
    print("Successful.")

    # Load model. Setup restores previous state if resuming training.
    print("-" * 79 + "\nLoading model...")
    model = create_model(configuration)
    model = setup_model(model)
    print("Successful.")

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
    with tqdm(total=len(track_paths)) as pbar:
        for track_path in track_paths:
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
            chunks = []
            offset = 0

            # Separate smaller windows of audio.
            while offset < max_frames:
                # Reshape and trim audio chunk.
                audio_chunk = mix_audio[:, offset : offset + length * sr]

                # Unsqueeze batch dimension if not already batched.
                if audio_chunk.dim() == 2:
                    audio_chunk = audio_chunk.unsqueeze(0)

                # Separate audio.
                estimate = model.separate(audio_chunk)

                # Trim end by padding amount.
                estimate = estimate[..., :-padding]
                chunks.append(estimate)

                # Update current frame position.
                offset = offset + step_size - padding
                if offset + length * sr >= max_frames:
                    break

            # Stitch chunks to create full source estimate.
            full_estimate = torch.cat(chunks, dim=2).reshape(
                (mix_audio.shape[0], -1)
            ).cpu().numpy()
            mix_trimmed = mix_audio.cpu().numpy()[..., :full_estimate.shape[-1]]
            res_trimmed = mix_trimmed - full_estimate

            # Export audio.
            track_name = track_path.name
            track_dir = save_dir.joinpath(track_name)
            track_dir.mkdir(parents=True, exist_ok=True)

            # Single target for now.
            label = model.target_labels[0]

            # Save outputs.
            wavfile.write(
                track_dir.joinpath(label).with_suffix(".wav"),
                rate=sr,
                data=full_estimate.T
            )
            wavfile.write(
                track_dir.joinpath("mixture").with_suffix(".wav"),
                rate=sr,
                data=mix_trimmed.T
            )
            if residual:
                wavfile.write(
                    track_dir.joinpath("residual").with_suffix(".wav"),
                    rate=sr,
                    data=res_trimmed.T
                )
            pbar.update(1)
