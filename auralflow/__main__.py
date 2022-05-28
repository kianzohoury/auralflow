# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git
import os
from argparse import ArgumentParser


from auralflow import datasets
from auralflow import losses
from auralflow import visualizer
from auralflow import models
from auralflow import trainer
from auralflow import utils
from auralflow import separate
from auralflow import train


__all__ = ["datasets", "losses", "visualizer", "models", "trainer", "utils"]

# def main():
#     pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Main script.")
    subparsers = parser.add_subparsers(dest="command")

    # Define configuration parser.
    config_parser = subparsers.add_parser(name="config")
    config_parser.add_argument(
        "folder_name",
        type=str,
        help="Model training folder name.",
        default="my_model"
    )
    config_parser.add_argument(
        "model_type", type=str, choices=models._models, help="Base model."
    )
    config_parser.add_argument(
        "--save",
        type=str,
        help="Location to save model training folder.",
        default=os.getcwd(),
        required=False
    )
    config_parser.add_argument(
        "--display",
        type=bool,
        help="Print configuration file.",
        default=False,
        required=False
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
        required=False
    )
    separator_parser.add_argument(
        "--residual",
        type=bool,
        help="Whether to include residual audio.",
        default=True,
        required=False
    )
    separator_parser.add_argument(
        "--length",
        type=int,
        help="Max cutoff length in seconds.",
        default=30,
        required=False
    )
    
    # Parse args.
    args = parser.parse_args()
    if args.command == "config":
        utils.pull_config_template(
            save_dir=args.save + "/" + args.folder_name
        )
        config = utils.load_config(args.folder_name + "/config.json")
        config["model_params"]["model_type"] = args.model_type
        config["model_params"]["model_name"] = args.folder_name
        utils.save_config(
            config, save_filepath=args.folder_name + "/config.json"
        )
    elif args.command == "train":
        config = utils.load_config(args.folder_name + "/config.json")
        config["dataset_params"]["dataset_path"] = args.dataset_path
        utils.save_config(
            config, save_filepath=args.folder_name + "/config.json"
        )
        train.main(config_filepath=args.folder_name + "/config.json")
    elif args.command == "separate":
        separate.main(
            config_filepath=args.folder_name + "/config.json",
            audio_filepath=args.audio_filepath,
            save_filepath=args.save,
            residual=args.residual,
            max_length=args.length
        )

