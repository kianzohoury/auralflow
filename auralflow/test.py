from csv import writer
from pathlib import Path

import asteroid.metrics
import librosa

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
    test_filepath = model.training_params["path"] + "/test"
    global_metrics = {
        "pesq": 0,
        "sar": 0,
        "sdr": 0,
        "si_sdr": 0,
        "sir": 0,
        "stoi": 0,
    }
    table_entries = []

    for track_name in Path(test_filepath).iterdir():
        label = model.target_labels[0]

        # Get stems.
        stems = separate_audio(
            model=model, filename=str(track_name), sr=44100
        )

        # Load target audio.
        target_audio, sr = librosa.load(
            str(track_name) + f"/{label}.wav", sr=44100
        )

        named_metrics = asteroid.metrics.get_metrics(
            mix=stems["mix"],
            clean=target_audio,
            estimate=stems["estimate"],
            sample_rate=sr,
            metrics_list="all",
            ignore_metrics_errors=True,
            average=True,
        )

        row = [track_name]
        for metric_label in sorted(named_metrics.keys()):
            global_metrics[metric_label] += named_metrics[metric_label]
            row.append(named_metrics[metric_label])
        table_entries.append(row)

    row = ["mean"]
    for metric_label in sorted(global_metrics.keys()):
        row.append(global_metrics[metric_label] / len(table_entries))
    table_entries.append(row)

    # Save as csv file.
    with open(save_filepath + "/metrics.csv", mode="w") as file:
        writer(csvfile=file)
