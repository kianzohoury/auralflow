# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from argparse import ArgumentParser, ArgumentError, Namespace
from auralflow import configurations
from pathlib import Path
from typing import List

# Define constants for parsing configurations.
CONSTRUCTION_LOSS_NAMES = ["l1", "l2"]
CRITERION_NAMES = [
    "component",
    "kl_div",
    "l1",
    "l2",
    "mask",
    "si_sdr",
    "rmse"
]
MASK_ACT_FN_NAMES = [
    "relu",
    "hardtanh",
    "tanh",
    "softmax",
    "prelu",
    "selu",
    "elu",
    "sigmoid"
]
MODEL_NAMES = [
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE"
]
TARGETS = ["bass", "drums", "other", "vocals"]


def _add_default_args(sub_parser: ArgumentParser, fields, **kwargs) -> None:
    """Adds default arguments to the command line parser."""
    for field in fields:
        if field.type is bool:
            sub_parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                default=False,
                const=True,
                nargs='?',
            )
        else:
            sub_parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                type=field.type,
                default=field.default,
                required=False,
                choices=kwargs.get(field.name, None)
            )

            
def parse():
    parser = ArgumentParser(
        description="Main script."
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # Define model configuration parser.
    config_parser = subparsers.add_parser(name="config")
    config_parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_NAMES,
        required=False, 
        default="SpectrogramNetLSTM",
        help="Base model."
    )
    for target in TARGETS:
        config_parser.add_argument(
            f"--{target}",
            required=False,
            default=False,
            const=True,
            nargs='?',
            help=f"Estimate {target}."
        )
    config_parser.add_argument(
        "--save",
        default=str(Path.cwd().joinpath("my_model")),
        required=False,
        help="Path to the folder which will store all files created."
    )
    config_parser.add_argument(
        "--display",
        required=False,
        default=False,
        const=True,
        nargs='?',
        help="Displays the model spec after its configuration file is created."
    )

    # Store default model configuration optional args.
    _add_default_args(
        config_parser,
        configurations.SpecModelConfig.defaults(),
        **{"mask_act_fn": MASK_ACT_FN_NAMES}
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
        help="Resumes model training from the checkpoint file.",
        default=False,
        const=True,
        nargs='?'
    )
    train_parser.add_argument(
        "--display",
        help="Displays the training parameters after the file is created.",
        default=False,
        const=True,
        nargs='?'
    )

    # Store default training configuration optional args.
    _add_default_args(
        train_parser,
        configurations.TrainingConfig.defaults(),
        **{
            "criterion": CRITERION_NAMES,
            "construction_loss": CONSTRUCTION_LOSS_NAMES
        }
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
        default=str(Path.cwd()),
        required=False,
    )
    separator_parser.add_argument(
        "--residual",
        help="Whether to include residual audio (if possible).",
        default=False,
        const=True,
        nargs='?',
    )
    separator_parser.add_argument(
        "--duration",
        type=int,
        help="Max duration in seconds.",
        default=210,
        required=False,
    )
    separator_parser.add_argument(
        "--sr",
        type=int,
        help="Sample rate.",
        default=44100,
        required=False,
    )
    separator_parser.add_argument(
        "--padding",
        type=int,
        help="Padding",
        default=200,
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

    # Define pretrained loader parser.
    pretrained_parser = subparsers.add_parser(name="pretrained")
    pretrained_parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_NAMES,
        required=False,
        default="SpectrogramNetLSTM",
        help="Base model."
    )
    pretrained_parser.add_argument(
        "--target",
        required=False,
        default="vocals"
    )
    pretrained_parser.add_argument(
        "--save",
        default=str(Path.cwd().joinpath("my_model")),
        required=False,
        help="Path to the folder which will store the pretrained weights."
    )
    args = parser.parse_args()
    return args


def parse_targets(args: Namespace) -> List[str]:
    targets = [target for target in TARGETS if getattr(args, target)]
    if not targets:
        raise ArgumentError(
            argument=None,
            message=f"Missing target source(s) to estimate: {TARGETS}."
        )
    return targets
