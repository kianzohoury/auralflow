# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch

from argparse import ArgumentParser
from torch import Tensor


def main(model, audio: Tensor):
    # Split audio in chunks.
    padding = model.dataset_params["hop_length"] // 2
    step_size = model.dataset_params.num_stft_frames
    max_frames = audio.shape[-1]

    # Store chunks.
    chunks = []

    offset = 0
    while offset < max_frames:
        # Reshape and trim audio chunk.
        audio_chunk = audio[:, offset : offset + step_size]

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

    # Stitch chunks to create full source estimate.
    full_estimate = torch.cat(
        chunks,
        dim=0,
    )
    return full_estimate


if __name__ == "__main__":
    parser = ArgumentParser(description="Source separation script.")
    parser.add_argument(
        "model_checkpoint", type=str, help="Path to a source separation model."
    )
    parser.add_argument(
        "audio_filepath", type=str, help="Path to an audio file."
    )
    args = parser.parse_args()
    model_path, audio_filepath = args.model_checkpoint, args.audio_filepath

    main(args.audio_filepath)
