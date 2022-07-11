# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import dataclasses
import os
import pprint

from argparse import ArgumentParser, ArgumentError

import torch.cuda

from auralflow.models import ALL_MODELS, AUDIO_MODELS, SPEC_MODELS
from auralflow.trainer import AudioModelConfig, SpecModelConfig, CriterionConfig, TrainingConfig, VisualsConfig
from auralflow.trainer import setup
from auralflow import separate
from auralflow import train
from auralflow import test
from auralflow.utils import save_config, load_config
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser(description="Main script.")
    subparsers = parser.add_subparsers(dest="command")

    # Define model configuration parser.
    config_parser = subparsers.add_parser(name="config")
    config_parser.add_argument(
        "model_type", type=str, choices=ALL_MODELS, help="Base model."
    )
    config_parser.add_argument(
        "-b",
        "--bass",
        action="store_true",
        required=False,
        help="Estimate bass."
    )
    config_parser.add_argument(
        "-d",
        "--drums",
        action="store_true",
        required=False,
        help="Estimate drums."
    )
    config_parser.add_argument(
        "-o",
        "--other",
        action="store_true",
        required=False,
        help="Estimate other/background."
    )
    config_parser.add_argument(
        "-v",
        "--vocals",
        action="store_true",
        required=False,
        help="Estimate vocals."
    )
    config_parser.add_argument(
        "--save",
        default=str(Path(os.getcwd(), "my_model").absolute()),
        required=False,
        help="Path to the folder which will store all files created."
    )
    config_parser.add_argument(
        "--display",
        action="store_true",
        required=False,
        help="Displays the model spec after its configuration file is created"
    )
    # Store default model configuration keyword args.
    for key, val in AudioModelConfig.defaults().items():
        if isinstance(val, bool):
            config_parser.add_argument(
                f"--{key.replace('_', '-')}",
                required=False,
                action="store_true"
            )
        else:
            config_parser.add_argument(
                f"--{key.replace('_', '-')}",
                default=val,
                required=False
            )

    # Define training parser.
    train_parser = subparsers.add_parser(name="train")
    train_parser.add_argument(
        "folder_name", type=str, help="Path to model training folder."
    )
    train_parser.add_argument(
        "dataset_path", type=str, help="Path to a valid dataset."
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
    train_parser.add_argument(
        "--resume",
        help="Resumes training from the previous state.",
        required=False,
        action="store_true"
    )
    # Store default criterion configuration keyword args.
    for key, val in CriterionConfig.defaults().items():
        if isinstance(val, bool):
            train_parser.add_argument(
                f"--{key.replace('_', '-')}",
                required=False,
                action="store_true"
            )
        else:
            train_parser.add_argument(
                f"--{key.replace('_', '-')}",
                default=val,
                required=False
            )
    # Store default training configuration keyword args.
    for key, val in TrainingConfig.defaults().items():
        if isinstance(val, bool):
            train_parser.add_argument(
                f"--{key.replace('_', '-')}",
                required=False,
                action="store_true"
            )
        elif key not in ["max_tracks", "max_samples", "dataset_path"]:
            train_parser.add_argument(
                f"--{key.replace('_', '-')}",
                default=val,
                required=False
            )
    # Store default visualization configuration keyword args.
    for key, val in VisualsConfig.defaults().items():
        if isinstance(val, bool):
            train_parser.add_argument(
                f"--{key.replace('_', '-')}",
                required=False,
                action="store_true"
            )
        else:
            train_parser.add_argument(
                f"--{key.replace('_', '-')}",
                default=val,
                required=False
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

        # Get target labels.
        if not (args.bass or args.drums or args.other or args.vocals):
            raise ArgumentError(
                argument=None,
                message=(
                    "Missing target source(s) to estimate. Specify targets "
                    "with flags: -b (bass), -d (drums), -o (other), "
                    "-v (vocals)."
                )
            )
        else:
            targets = []
            targets.extend(["bass"] if args.__dict__.pop("bass") else [])
            targets.extend(["drums"] if args.__dict__.pop("drums") else [])
            targets.extend(["other"] if args.__dict__.pop("other") else [])
            targets.extend(["vocals"] if args.__dict__.pop("vocals") else [])

        # Create model configuration from args.
        model_config = setup._create_model_config(
            model_type=args.__dict__.pop("model_type"),
            targets=targets,
            **args.__dict__
        )

        # Save the model configuration file within the specified directory.
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_config.save(filepath=str(save_dir.joinpath("model.json")))

        # Display model config as a table.
        if args.display:
            print(model_config)

    elif args.command == "train":
        print("Reading configuration files...")
        save_dir = Path(args.folder_name)

        # Load model configuration.
        model_config = setup._load_model_config(
            filepath=str(save_dir.joinpath("model.json"))
        )

        # Create loss criterion configuration.
        if isinstance(model_config, SpecModelConfig):
            input_type = "spectrogram"
        else:
            input_type = "audio"
        criterion_config = setup.CriterionConfig.from_dict(
            input_type=input_type, **args.__dict__
        )

        # Set default logging directory path if tensorboard is enabled.
        if not args.tensorboard:
            logging_dir = None
        else:
            logging_dir = str(save_dir.joinpath("runs"))

        # Create visualization configuration.
        visuals_config = VisualsConfig.from_dict(
            logging_dir=logging_dir,
            **args.__dict__
        )

        # Create trainer configuration.
        training_config = TrainingConfig.from_dict(
            criterion_config=criterion_config,
            visuals_config=visuals_config,
            checkpoint=str(save_dir.joinpath("checkpoint.pth")),
            device="cuda" if torch.cuda.is_available() else "cpu",
            **args.__dict__
        )

        # Save the training configuration in the same directory.
        training_config.save(filepath=str(save_dir.joinpath("trainer.json")))

        if not args.resume:
            # Delete existing metadata files.
            for metadata_file in list(save_dir.glob("*.pickle")):
                metadata_file.unlink(missing_ok=True)
            # Delete existing checkpoint.
            Path(save_dir.joinpath("checkpoint.pth")).unlink(missing_ok=True)

        train.main(
            model_config=model_config,
            save_dir=str(save_dir),
            training_config=training_config,
            dataset_path=args.dataset_path,
            max_num_tracks=args.max_tracks,
            max_num_samples=args.max_samples,
            resume=args.resume
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
