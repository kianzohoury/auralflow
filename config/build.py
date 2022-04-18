import config.utils
import sys
import torch.nn as nn
from pathlib import Path
import ruamel.yaml
from yaml import YAMLError
from config.constants import REQUIRED_MODEL_KEYS, BASE_MODELS
from AudioFolder.datasets import AudioFolder
from ruamel.yaml.error import MarkedYAMLError
import textwrap
from ruamel.yaml.scanner import ScannerError
import models
from pprint import pprint
import time

yaml_parser = ruamel.yaml.YAML(typ='safe', pure=True)
session_logs_file = Path(__file__).parent / 'session_logs.yaml'


class BuildFailure(Exception):
    def __init__(self, error):
        self.error = error
        self.message = ("Cannot build model. Check configuration file or"
                        "arguments to build().")

    def __repr__(self):
        return f"{self.error}. {self.message}"


def compress_keys(config_dict: dict) -> dict:
    """Helper function to merge and reformat config dict keys."""
    if not config_dict:
        return {}
    result = {}
    for k1, v1 in config_dict.items():
        if isinstance(v1, dict):
            flattened = compress_keys(v1)
            for k2 in flattened:
                result[f"{k1}_{k2}".replace('-', '_')] = flattened[k2]
        else:
            result[k1.replace('-', '_')] = v1
    return result


def get_all_config_contents(model_dir: Path) -> dict:
    """Gets all configuration file contents from a model's folder.

    Args:
        model_dir (Path): Model directory to read from.

    Returns:
        (dict): The training session configuration dictionary containing
            model, data and training settings.

    Raises:
        YAMLError: If a configuration file cannot be read.
    """
    try:
        frmt_config_data = {}
        for config_file in model_dir.glob("*.yaml"):
            config_data = yaml_parser.load(config_file)
            for label in config_data.keys():
                frmt_config_data[label] = compress_keys(config_data[label])
        # Make sure only keys are 'model', 'data' and 'training'.
        if set(frmt_config_data.keys()) == {'data', 'model', 'training'}:
            return frmt_config_data
    except MarkedYAMLError as e:
        raise YAMLError(f"Cannot read files {e.context}. Check the formatting"
                        " of your configuration files, or create a new model to" 
                        " restore the configuration files.")


def build_model(config_dict: dict) -> nn.Module:
    """Implements a PyTorch model from the configuration files.

    Args:
        config_dict (dict): Training configuration dictionary.

    Returns:
        (nn.Module): Pytorch model.
    Raises:
        BuildFailure: Failed to execute Pytorch module creation.
    """
    base_model = config_dict['model']['base_model']
    build_config = {}
    for _, config_map in config_dict.items():
        for param_name, param_val in config_map.items():
            if param_name in REQUIRED_MODEL_KEYS:
                build_config[param_name] = param_val
    try:
        return BASE_MODELS[base_model](**build_config)
    except Exception as error:
        raise BuildFailure(error)


def build_audio_folder(config_dict: dict, dataset_dir: Path):
    """Creates an AudioFolder for sampling audio from the dataset.

    Args:
        config_dict (dict): Training configuration dictionary.
        dataset_dir (Path): Path to the data set folder.

    Returns:
        (AudioFolder): An AudioFolder iterable dataset.

    Raises:
        FileNotFoundError: Path to dataset could not be found.
    """
    data_config = config_dict['data']
    try:
        return AudioFolder(str(dataset_dir), **data_config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot load {str(dataset_dir)} into an "
                                "AudioFolder. Check the directory's path.")
