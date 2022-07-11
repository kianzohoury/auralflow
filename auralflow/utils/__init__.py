# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import json
import os
import shutil

from pathlib import Path
from typing import Dict, Optional, Union

from prettytable import PrettyTable


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


def load_config(save_filepath: str) -> Dict:
    """Loads configuration data given its filepath.

    Args:
        save_filepath (str): Path to load config file from.

    Raises:
        IOError: Raised if the configuration file cannot be loaded.
    """
    try:
        with open(save_filepath, "r") as config_file:
            config_data = json.load(fp=config_file)
            return config_data
    except IOError as error:
        raise error
