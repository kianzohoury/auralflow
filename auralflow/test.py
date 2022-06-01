
from pathlib import Path

import asteroid.metrics
import librosa
import csv

from auralflow.models import create_model, setup_model
from auralflow.separate import separate_audio
from auralflow.utils import load_config


def main(config_filepath: str, save_filepath: str) -> None:
    """Tests separated audio and saves metrics as a csv file."""

    # Load configuration file.
    print("Reading configuration file...")
    configuration = load_config(config_filepath)
    print("  Successful.")

    # Load model. Setup restores previous state if resuming training.
    print("Loading model...")
    model = create_model(configuration)
    model = setup_model(model)
    print("  Successful.")

    # Path to test set.
    test_filepath = model.dataset_params["dataset_path"] + "/test"
    global_metrics = {
        "pesq": 0,
        "sar": 0,
        "sdr": 0,
        "si_sdr": 0,
        "stoi": 0,
    }
    table_entries = []

    print("Testing model...")
    for track_name in Path(test_filepath).iterdir():
        label = model.target_labels[0]

        # Load target audio.
        target_audio, sr = librosa.load(
            str(track_name) + f"/{label}.wav", sr=44100
        )

        # Get stems.
        stems = separate_audio(
            model=model,
            filename=str(track_name),
            sr=44100,
            duration=4
        )

        max_frames = stems["estimate"].shape[-1]

        named_metrics = asteroid.metrics.get_metrics(
            mix=stems["mix"][..., :max_frames].cpu().numpy(),
            clean=target_audio[..., :max_frames],
            estimate=stems["estimate"].cpu().numpy(),
            sample_rate=sr,
            metrics_list="all",
            ignore_metrics_errors=True,
            average=True,
        )

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
