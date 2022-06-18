# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import json
import shutil
import torch

from auralflow.models import SeparationModel
from pathlib import Path

__all__ = [
    "copy_config_template",
    "load_config",
    "save_config",
    "save_object",
    "load_object",
]

config_template_path = Path(__file__).parents[2].joinpath("config.json")


def copy_config_template(save_dir: str) -> None:
    """Clones default ``config.json`` file to a save directory.

    Args:
        save_dir (str): Path to copy ``config.json`` to.
    """
    Path(save_dir).mkdir(exist_ok=True)
    save_filepath = save_dir + "/config.json"
    shutil.copy(src=str(config_template_path), dst=save_filepath)


def load_config(config_filepath: str) -> dict:
    """Loads the contents of a configuration file into a dictionary.

    Args:
        config_filepath (str): Path to the configuration file.

    Returns:
        dict: Configuration data.

    Raises:
        IOError: Raised if the file cannot be read.
    """
    try:
        with open(config_filepath) as config_file:
            return json.load(config_file)
    except IOError as error:
        raise error


def save_config(config: dict, save_filepath: str) -> None:
    """Saves configuration data to the given filepath.

    Args:
        config (dict): Configuration data.
        save_filepath (str): Path to save.

    Raises:
        IOError: Raised if the configuration file cannot be saved.
    """
    try:
        with open(save_filepath, "w") as config_file:
            return json.dump(config, config_file, indent=4)
    except IOError as error:
        raise error


def _add_checkpoint_tag(filename: str, obj_name: str, global_step: int) -> str:
    """Attaches a suffix to the checkpoint filename depending on the object."""
    if obj_name == "model":
        filename += f"_{global_step}.pth"
    elif obj_name == "optimizer":
        filename += f"_optimizer_{global_step}.pth"
    elif obj_name == "scheduler":
        filename += f"_scheduler_{global_step}.pth"
    elif obj_name == "grad_scaler":
        filename += f"_autocast_{global_step}.pth"
    return filename


def save_object(
    model: SeparationModel, obj_name: str, global_step: int
) -> None:
    """Saves object state as .pth file under the checkpoint directory.

    Args:
        model (SeparationModel): Separation model.
        obj_name (str): Object to save.
        global_step (int): Global step.

    Raises:
        OSError: Raised if object cannot be saved.
    """
    filename = f"{model.checkpoint_path}/{model.model_name}"
    if not model.training_params["silent_checkpoint"]:
        print(f"Saving {obj_name}...")
    # Get object-specific filename.
    filename = _add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )
    if hasattr(model, obj_name):
        # Retrieve object's state.
        if obj_name == "model":
            state_dict = getattr(model, obj_name).cpu().state_dict()
            # Transfer model back to GPU if applicable.
            model.model.to(model.device)
        else:
            state_dict = getattr(model, obj_name).state_dict()
        try:
            # Save object's state to filename.
            torch.save(state_dict, f=filename)
            if not model.training_params["silent_checkpoint"]:
                print(f"  Successful.")
        except OSError as error:
            print(f"  Failed.")
            raise error


def load_object(
    model: SeparationModel, obj_name: str, global_step: int
) -> None:
    """Loads object's state and and attaches it to the model.

    Args:
        model (SeparationModel): Separation model.
        obj_name (str): Object to save.
        global_step (int): Global step.

    Raises:
        OSError: Raised if object cannot be loaded.
    """
    filename = f"{model.checkpoint_path}/{model.model_name}"
    print(f"Loading {obj_name}...")
    # Get object-specific filename.
    filename = _add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )
    try:
        # Try to read object state from the given file.
        state_dict = torch.load(filename, map_location=model.device)
    except (OSError, FileNotFoundError) as error:
        print("  Failed.")
        raise error
    if hasattr(model, obj_name):
        # Load state into object.
        getattr(model, obj_name).load_state_dict(state_dict)
        print(f"  Successful.")
