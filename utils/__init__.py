# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import json


__all__ = ["load_config"]


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
        with open(save_filepath) as config_file:
            return json.dump(config, config_file)
    except IOError as error:
        raise error
