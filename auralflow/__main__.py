# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import os

from argparse import ArgumentParser
from auralflow import models
from auralflow import utils
from auralflow import separate
from auralflow import train
from auralflow import test
from pathlib import Path
from prettytable import PrettyTable


_config_optionals = {
    "--normalize-input": bool,
    "--normalize-output": bool,
    "--mask-activation": str,
    "--hidden-channels": int,
    "--dropout-p": float,
    "--leak-factor": float,
    "--targets": str,
    "--max-num-tracks": int,
    "--max-num-samples": int,
    "--num-channels": int,
    "--sample-length": int,
    "--sample-rate": int,
    "--num-fft": int,
    "--window-size": int,
    "--hop-length": int,
    "--max-epochs": int,
    "--batch-size": int,
    "--lr": float,
    "--criterion": str,
    "--max-lr-steps": int,
    "--stop-patience": int,
    # "--view-spectrogram": bool,
    # "--view-waveform": bool,
    # "--view-gradient": bool,
    # "--play-audio": bool,
    # "--save-image": bool,
    # "--save-audio": bool,
    "--save-frequency": int,
}


if __name__ == "__main__":
    parser = ArgumentParser(description="Main script.")
    subparsers = parser.add_subparsers(dest="command")

    # Define configuration parser.
    config_parser = subparsers.add_parser(name="config")
    config_parser.add_argument(
        "folder_name",
        type=str,
        help="Model training folder name.",
        default="my_model",
    )
    config_parser.add_argument(
        "model_type", type=str, choices=models.model_names, help="Base model."
    )
    config_parser.add_argument(
        "--save",
        type=str,
        help="Location to save model training folder.",
        default=os.getcwd(),
        required=False,
    )
    config_parser.add_argument(
        "--display",
        help="Print configuration file.",
        action="store_true",
        required=False,
    )

    for optional_key, optional_type in _config_optionals.items():
        if optional_type is bool:
            config_parser.add_argument(
                optional_key, action="store_true", required=False
            )
        else:
            config_parser.add_argument(
                optional_key, type=optional_type, required=False
            )

    # Define training parser.
    train_parser = subparsers.add_parser(name="train")
    train_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    train_parser.add_argument(
        "dataset_path", type=str, help="Path to dataset."
    )

    # Define separator parser.
    separator_parser = subparsers.add_parser(name="separate")
    separator_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    separator_parser.add_argument(
        "audio_filepath", type=str, help="Path to an audio file or folder."
    )
    separator_parser.add_argument(
        "--save",
        type=str,
        help="Location to save separated audio.",
        default=os.getcwd(),
        required=False,
    )
    separator_parser.add_argument(
        "--residual",
        help="Whether to include residual audio.",
        action="store_true",
        required=False,
    )
    separator_parser.add_argument(
        "--duration",
        type=int,
        help="Max duration in seconds.",
        default=30,
        required=False,
    )

    # Define training parser.
    test_parser = subparsers.add_parser(name="test")
    test_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    test_parser.add_argument(
        "--duration",
        type=int,
        help="Max duration in seconds.",
        default=30,
        required=False,
    )
    test_parser.add_argument(
        "--max-tracks",
        type=int,
        help="Max number of tracks to test.",
        default=20,
        required=False,
    )
    test_parser.add_argument(
        "--resample-rate",
        type=int,
        help="Reduce sample rate.",
        default=44100,
        required=False,
    )

    # Parse args.
    args = parser.parse_args()
    if args.command == "config":
        save_dir = str(Path(args.save + "/" + args.folder_name).absolute())
        utils.copy_config_template(save_dir=save_dir)
        config = utils.load_config(save_dir + "/config.json")
        config["model_params"]["model_type"] = args.model_type
        config["model_params"]["model_name"] = args.folder_name
        config["model_params"]["save_dir"] = save_dir

        for optional_key, val in args.__dict__.items():
            if val is None:
                continue
            elif optional_key in config["model_params"]:
                config["model_params"][optional_key] = val
            elif optional_key in config["dataset_params"]:
                config["dataset_params"][optional_key] = val
            elif optional_key in config["training_params"]:
                config["training_params"][optional_key] = val
            elif optional_key in config["visualizer_params"]:
                config["visualizer_params"][optional_key] = val
        if args.display:
            for group_label, group in config.items():
                param_table = PrettyTable(["Parameter", "Value"])
                param_table.align = "l"
                param_table.title = group_label
                param_table.min_width = 21
                p_labels, p_vals = [], []
                for param_label, param in group.items():
                    p_labels.append(param_label)
                    p_vals.append([param])
                    param_table.add_row([param_label, param])
                print(param_table)
        utils.save_config(config, save_filepath=save_dir + "/config.json")

    elif args.command == "train":
        config = utils.load_config(args.folder_name + "/config.json")
        config["dataset_params"]["dataset_path"] = args.dataset_path
        config["training_params"]["training_mode"] = True
        utils.save_config(
            config, save_filepath=args.folder_name + "/config.json"
        )
        train.main(config_filepath=args.folder_name + "/config.json")
    elif args.command == "separate":
        config = utils.load_config(args.folder_name + "/config.json")
        config["training_params"]["training_mode"] = False
        utils.save_config(
            config, save_filepath=args.folder_name + "/config.json"
        )
        separate.main(
            config_filepath=args.folder_name + "/config.json",
            audio_filepath=args.audio_filepath,
            save_filepath=args.save,
            residual=args.residual,
            duration=args.duration,
        )
    elif args.command == "test":
        config = utils.load_config(args.folder_name + "/config.json")
        config["training_params"]["training_mode"] = False
        utils.save_config(
            config, save_filepath=args.folder_name + "/config.json"
        )
        test.main(
            config_filepath=args.folder_name + "/config.json",
            save_filepath=args.folder_name,
            duration=args.duration,
            max_tracks=args.max_tracks,
            resample_rate=args.resample_rate,
        )
