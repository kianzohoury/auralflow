# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import json
import torch


__all__ = ["load_config", "save_config", "load_object", "save_object"]


def load_config(config_filepath: str):
    """Loads a .json configuration file given a filepath."""
    try:
        with open(config_filepath) as config_file:
            return json.load(config_file)
    except IOError as error:
        raise error


def save_config(config: dict, save_filepath: str):
    """Saves configuration data to a .json file at a given location."""
    try:
        with open(save_filepath, "w") as config_file:
            return json.dump(config, config_file)
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


def save_object(model_wrapper, obj_name: str, global_step: int) -> None:
    """Saves object state as .pth file under the checkpoint directory."""
    filename = f"{model_wrapper.checkpoint_path}/{model_wrapper.model_name}"

    # Get object-specific filename.
    filename = _add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )
    if hasattr(model_wrapper, obj_name):
        # Retrieve object's state.
        if obj_name == "model":
            state_dict = getattr(model_wrapper, obj_name).cpu().state_dict()
            # Transfer model back to GPU if applicable.
            model_wrapper.model.to(model_wrapper.device)
        else:
            state_dict = getattr(model_wrapper, obj_name).state_dict()
        try:
            # Save object's state to filename.
            torch.save(state_dict, f=filename)
            if not model_wrapper.training_params["silent_checkpoint"]:
                print(f"Successfully saved {obj_name}.")
        except OSError as error:
            print(f"Failed to save {obj_name} state.")
            raise error


def load_object(model_wrapper, obj_name: str, global_step: int) -> None:
    """Loads object and attaches it to model_wrapper and its device."""
    filename = f"{model_wrapper.checkpoint_path}/{model_wrapper.model_name}"

    # Get object-specific filename.
    filename = _add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )
    try:
        # Try to read object state from the given file.
        state_dict = torch.load(filename, map_location=model_wrapper.device)
    except (OSError, FileNotFoundError) as error:
        print(f"Failed to load {obj_name} state.")
        raise error
    if hasattr(model_wrapper, obj_name):
        # Load state into object.
        getattr(model_wrapper, obj_name).load_state_dict(state_dict)
        print(f"Loaded {obj_name} successfully.")
