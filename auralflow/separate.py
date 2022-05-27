# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from argparse import ArgumentParser

import librosa
import torch
from scipy.io import wavfile

from auralflow.models import create_model, setup_model
from auralflow.utils import load_config


def main(config_filepath: str, audio_filepath: str, save_filepath: str):
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

    # Load audio.
    audio, sr = librosa.load(audio_filepath, sr=44100)
    audio = torch.from_numpy(audio).unsqueeze(0)
    length = model.dataset_params["sample_length"]

    # Split audio into chunks.
    padding = 200
    step_size = length * sr
    max_frames = audio.shape[-1]

    # Store chunks.
    chunks = []
    offset = 0

    print("Separating...")
    while offset < max_frames:
        # Reshape and trim audio chunk.
        audio_chunk = audio[:, offset : offset + length * sr]

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
    full_estimate = torch.cat(chunks, dim=2).reshape(audio.shape)
    # Export audio.
    wavfile.write(save_filepath, rate=sr, data=full_estimate.cpu().numpy())


if __name__ == "__main__":
    parser = ArgumentParser(description="Source separation script.")
    parser.add_argument(
        "config_filepath", type=str, help="Path to a configuration file."
    )
    parser.add_argument(
        "audio_filepath", type=str, help="Path to an audio file."
    )
    parser.add_argument(
        "save_filepath", type=str, help="Path to save audio to."
    )
    args = parser.parse_args()
    main(args.config_filepath, args.audio_filepath, args.save_filepath)
