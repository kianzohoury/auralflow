# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch
import librosa
from scipy.io import wavfile 

from argparse import ArgumentParser
from auralflow.models import load_pretrained_model


def main(checkpoint_dir: str, audio_path: str, save_path):

    model = load_pretrained_model(checkpoint_dir)

    audio, sr = librosa.load(audio_path, sr=44100)
    audio = torch.from_numpy(audio).unsqueeze(0)
    print(audio.shape)
    length = model.dataset_params["sample_length"]


    # Split audio in chunks.
    padding = model.dataset_params["hop_length"] // 2
    step_size = model.n_fft_bins
    step_size = length * sr
    max_frames = audio.shape[-1]

    # Store chunks.
    chunks = []

    offset = 0
    while offset < max_frames:
        # Reshape and trim audio chunk.
        audio_chunk = audio[:, offset : offset + length * sr]

        # Unsqueeze batch dimension if not already batched.
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.unsqueeze(0)

        print(audio_chunk.shape)

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
    full_estimate = torch.cat(
        chunks,
        dim=2,
    ).squeeze(0).squeeze(0)

    print(full_estimate.cpu().numpy().shape)

    wavfile.write(save_path, sr, full_estimate.cpu().numpy())
    return full_estimate


if __name__ == "__main__":
    parser = ArgumentParser(description="Source separation script.")
    parser.add_argument(
        "checkpoint_dir", type=str, help="Path to a source separation model."
    )
    parser.add_argument(
        "audio_filepath", type=str, help="Path to an audio file."
    )
    parser.add_argument(
        "save", type=str, help="Path to output."
    )
    args = parser.parse_args()
    model_path, audio_filepath = args.checkpoint_dir, args.audio_filepath
    output_path = args.save

    main(model_path, args.audio_filepath, output_path)
