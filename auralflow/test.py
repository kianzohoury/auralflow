from pathlib import Path

# import asteroid.metrics

import numpy as np
import torch
import csv

from auralflow.trainer import setup
from auralflow.separate import separate_audio
from auralflow.utils import load_config
from torchaudio.transforms import Resample
from auralflow.losses import si_sdr_loss
import matplotlib.pyplot as plt


def main(
    config_filepath: str,
    save_filepath: str,
    duration: int = 30,
    max_tracks: int = 20,
    resample_rate: int = 44100,
) -> None:
    """Tests separated audio and saves metrics as a csv file."""

    # Load configuration file.
    print("Reading configuration file...")
    configuration = load_config(config_filepath)
    print("  Successful.")

    # Load model. Setup restores previous state if resuming training.
    model = setup._build_from_config(*configuration)
    # model = setup_model(model)

    # Path to test set.
    test_filepath = model.dataset_params["dataset_path"] + "/test"
    global_metrics = {
        "sar": 0,
        "sdr": 0,
        "si_sdr": 0,
        "stoi": 0,
    }
    table_entries = []

    print("Testing model...")
    for track_name in list(Path(test_filepath).iterdir())[:max_tracks]:
        label = model.targets[0]

        # Get stems.
        stems = separate_audio(
            model=model, filename=str(track_name), sr=44100, duration=duration
        )

        max_frames = stems["estimate"].shape[-1]

        # Load target audio.
        # target_audio, sr = librosa.load(
        #     str(track_name) + f"/{label}.wav", sr=44100, dtype=np.float32
        # )

        # # Reduce sample rate.
        # resampler = Resample(
        #     orig_freq=44100, new_freq=resample_rate, dtype=torch.float32
        # )
        # mix = resampler(stems["mix"][..., :max_frames].cpu()).numpy()
        # estimate = resampler(stems["estimate"].cpu()).numpy()
        # target = resampler(
        #     torch.from_numpy(target_audio[..., :max_frames]).float()
        # ).reshape(estimate.shape).numpy()

        mix = stems["mix"][..., :max_frames].cpu().numpy()
        estimate = stems["estimate"].cpu().numpy()
        # target = torch.from_numpy(
        #     target_audio[..., :max_frames]
        # ).float().reshape(estimate.shape).numpy()
        target = target_audio[:max_frames].reshape(estimate.shape)

        si_sdr = si_sdr_loss(
            torch.from_numpy(estimate).float().unsqueeze(0),
            torch.from_numpy(target).float().unsqueeze(0)
        )

        named_metrics = {
            "si_sdr": -si_sdr.item()
        }

        # named_metrics = asteroid.metrics.get_metrics(
        #     mix=mix,
        #     clean=target,
        #     estimate=estimate,
        #     sample_rate=sr,
        #     compute_permutation=False,
        #     ignore_metrics_errors=True,
        #     average=True,
        #     filename=track_name.name,
        #     metrics_list=["sar", "sdr", "si_sdr", "stoi"],
        # )

        row = {"track_name": track_name.name}
        for metric_label, val in named_metrics.items():
            if metric_label in global_metrics and val is not None:
                global_metrics[metric_label] += val
                row[metric_label] = val
        table_entries.append(row)

    row = {"track_name": "average"}
    for metric_label, val in global_metrics.items():
        row[metric_label] = global_metrics[metric_label] / len(table_entries)
    table_entries.append(row)

    column_labels = ["track_name"] + sorted(global_metrics.keys())

    # Save as csv file.
    with open(save_filepath + "/metrics.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=column_labels)
        csv_writer.writeheader()
        csv_writer.writerows(table_entries)
