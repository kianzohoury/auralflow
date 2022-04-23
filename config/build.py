import torch.nn as nn
import sys
from pprint import pprint

from pathlib import Path
import ruamel.yaml
from yaml import YAMLError

from config.constants import REQUIRED_MODEL_KEYS, BASE_MODELS
from audio_folder.datasets import AudioFolder
from ruamel.yaml.error import MarkedYAMLError
from models.layers import StackedBlock
import textwrap
from ruamel.yaml.scanner import ScannerError
import models

from typing import List, Optional
import time

yaml_parser = ruamel.yaml.YAML(typ='safe', pure=True)
session_logs_file = Path(__file__).parent / 'session_logs.yaml'


class BuildFailure(Exception):
    def __init__(self, error):
        self.error = error
        self.message = ("Cannot build model. Check configuration file for"
                        "missing or invalid arguments.")

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


def build_model(config_data: dict) -> nn.Module:
    """Implements a PyTorch model from the configuration files.

    Args:
        config_data (dict): Model, training and dataset configuration data.

    Returns:
        (nn.Module): Pytorch model.

    Raises:
        BuildFailure: Failed to execute Pytorch module creation.
    """
    model_config = compress_keys(config_data['model'])
    dataset_config = compress_keys(config_data['dataset'])
    base_model = model_config.pop('base_model')
    build_instructions = {**model_config, **dataset_config}
    for config_key in build_instructions.copy():
        if config_key not in REQUIRED_MODEL_KEYS:
            build_instructions.pop(config_key)
    try:
        return BASE_MODELS[base_model](**build_instructions)
    except Exception as error:
        raise BuildFailure(error)


def build_audio_folder(config_data: dict, dataset_dir: Optional[Path] = None):
    """Creates an AudioFolder for sampling audio from the dataset.

    Args:
        config_data (dict): Model, training and dataset configuration data.
        dataset_dir (Path or None): Path to the dataset folder if it's not
            specified within the dataset configuration file.

    Returns:
        (audio_folder): An audio_folder iterable dataset.

    Raises:
        FileNotFoundError: Path to dataset could not be found.
    """
    dataset_config = compress_keys(config_data['dataset'])
    audio_transform = {}
    for key in ['num_fft', 'window_size', 'hop_length']:
        try:
            audio_transform[key] = dataset_config.pop(key)
        except KeyError as error:
            raise BuildFailure(error)
    dataset_config['transform'] = audio_transform
    try:
        dataset_dir = dataset_dir or Path(dataset_config.pop('path'))
        dataset_config['path'] = str(dataset_dir)
        return AudioFolder(**dataset_config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot load {str(dataset_dir)} into an "
                                "AudioFolder. Check the directory's path.")


def build_layers(config_dict: dict) -> List[dict]:
    d1 = yaml_parser.load(Path('/Users/Kian/Desktop/auralflow/config/unet/unet_base.yaml'))
    d2 = yaml_parser.load(Path('/Users/Kian/Desktop/auralflow/config/data_config.yaml'))
    # pprint(d1 | d2)
    x = build_model(d1 | d2)
    return [{}]
#
# def parse_layers(config_dict: dict):
