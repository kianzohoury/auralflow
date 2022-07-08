# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import os

from argparse import ArgumentParser
from auralflow.models import ALL_MODELS, AUDIO_MODELS, SPEC_MODELS
from auralflow.trainer import AudioModelConfig, SpecModelConfig, _parse_to_config, TrainingConfig
from auralflow import separate
from auralflow import train
from auralflow import test
from auralflow.utils import save_config, load_config
from pathlib import Path
from prettytable import PrettyTable


if __name__ == "__main__":
    parser = ArgumentParser(description="Main script.")
    subparsers = parser.add_subparsers(dest="command")

    # Define configuration parser.
    config_parser = subparsers.add_parser(name="config")
    config_parser.add_argument(
        "model_type", type=str, choices=ALL_MODELS, help="Base model."
    )
    config_parser.add_argument(
        "--save",
        type=str,
        help="Relative path of the folder which will store all created files.",
        default=str(Path(os.getcwd(), "my_model").absolute()),
        required=False,
    )
    config_parser.add_argument(
        "--display",
        help="Print model configuration settings.",
        action="store_true",
        required=False,
    )

    # # Fill keyword args with defaults.
    # # Only consider spectrogram-based models for now.
    # for optional_key, optional_type in _config_optionals.items():
    #     if optional_type is bool:
    #         config_parser.add_argument(
    #             optional_key, action="store_true", required=False
    #         )
    #     else:
    #         config_parser.add_argument(
    #             optional_key, type=optional_type, required=False
    #         )

    # Define training parser.
    train_parser = subparsers.add_parser(name="train")
    train_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    train_parser.add_argument(
        "dataset_path", type=str, help="Path to a dataset."
    )
    train_parser.add_argument(
        "--max-tracks",
        type=int,
        help="Max number of tracks to use from the given dataset.",
        default=80,
        required=False,
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        help="Max number of samples to generate from the pool of tracks.",
        default=10000,
        required=False,
    )

    # Define separator parser.
    separator_parser = subparsers.add_parser(name="separate")
    separator_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    separator_parser.add_argument(
        "audio_filepath", type=str, help="Path to an audio file/folder."
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
        help="Whether to include residual audio (if possible).",
        action="store_true",
        required=False,
    )
    separator_parser.add_argument(
        "--duration",
        type=int,
        help="Max duration in seconds.",
        default=210,
        required=False,
    )

    # Define test parser.
    test_parser = subparsers.add_parser(name="test")
    test_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    test_parser.add_argument(
        "--duration",
        type=int,
        help="Max duration in seconds.",
        default=210,
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
        "--sr",
        type=int,
        help="Sample rate.",
        default=44100,
        required=False,
    )

    # Parse args.
    args = parser.parse_args()
    if args.command == "config":

        model_type = args.model_type
        save_dir = Path(os.getcwd(), args.save)

        model_config = _parse_to_config(model_type=model_type, **args.__dict__)

        # Make folder and save the configuration file inside it.
        save_dir.mkdir(parents=True, exist_ok=True)

        save_config(
            config=model_config.__dict__,
            save_filepath=str(save_dir.joinpath("model.json"))
        )

        # # Display model config as a table.
        # if args.display:
        #     param_table = PrettyTable(["Parameter", "Value"])
        #     param_table.align = "l"
        #     param_table.title = "Model Configuration"
        #     param_table.min_width = 21
        #     p_labels, p_vals = [], []
        #     for param_label, param in config["model_params"]:
        #         p_labels.append(param_label)
        #         p_vals.append([param])
        #         param_table.add_row([param_label, param])
        #     print(param_table)

    elif args.command == "train":
        print("Reading configuration file...")

        save_dir = Path(os.getcwd(), args.save)
        model_config_data = load_config(
            save_filepath=str(save_dir.joinpath("model.json"))
        )
        model_config = _parse_to_config(
            model_type=model_config_data["model_type"],
            **model_config_data
        )

        training_config = TrainingConfig(**args.__dict__)
        train.main(
            model_config=model_config,
            training_config=training_config,
            dataset_path=args.dataset_path,
            max_num_tracks=args.max_tracks,
            max_num_samples=args.max_samples
        )

    elif args.command == "separate":
        config = load_config(args.folder_name)
        config["training_params"]["training_mode"] = False
        save_config(
            config, save_filepath=args.folder_name
        )
        separate.main(
            config_filepath=args.folder_name,
            audio_filepath=args.audio_filepath,
            save_filepath=args.save,
            residual=args.residual,
            duration=args.duration,
        )
    elif args.command == "test":
        config = load_config(args.folder_name)
        config["training_params"]["training_mode"] = False
        save_config(
            config, save_filepath=args.folder_name
        )
        test.main(
            config_filepath=args.folder_name,
            save_filepath=args.folder_name,
            duration=args.duration,
            max_tracks=args.max_tracks,
            resample_rate=args.resample_rate,
        )
