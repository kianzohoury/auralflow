import torch
import torch.nn as nn
import sys
from pprint import pprint

from pathlib import Path
import ruamel.yaml
import torchinfo
from yaml import YAMLError

from config.constants import REQUIRED_MODEL_KEYS, BASE_MODELS
from audio_folder.datasets import AudioFolder
from ruamel.yaml.error import MarkedYAMLError
from models.base_unet import BaseUNet
from transforms import get_data_shape
from collections import OrderedDict
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
        input_size = get_data_shape(
            batch_size=1,
            sample_rate=dataset_config['sample_rate'],
            sample_length=dataset_config['sample_length'],
            num_fft=dataset_config['num_fft'],
            window_size=dataset_config['window_size'],
            hop_length=dataset_config['hop_length'],
            num_channels=dataset_config['num_channels'],
            targets=build_instructions['targets']
        )
        num_bins, num_samples = input_size[1:3]
        build_instructions['num_bins'] = num_bins
        build_instructions['num_samples'] = num_samples
        encoder_block = process_block(
            build_instructions.pop('encoder'), block_type='encoder'
        )
        decoder_block = process_block(
            build_instructions.pop('decoder'), block_type='decoder'
        )
        build_instructions['encoder'] = encoder_block
        build_instructions['decoder'] = decoder_block
        model = BASE_MODELS[base_model](**build_instructions)

        # Test the network using via a forward pass.
        test_data = torch.rand(input_size[:-1])
        model(test_data)

        return model
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


def process_block(block_scheme: List, block_type: str = 'encoder') -> dict:
    """Processes raw autoencoder structures."""

    assert len(block_scheme) > 0, f"Must specify at least 1 layer."

    processed_block = []
    use_max_pool = False
    use_upsample = False
    num_convs = 0
    num_transpose = 0

    for layer in block_scheme:
        if len(layer) == 1:
            layer.append(None)

    for i, (layer_type, param) in enumerate(block_scheme):
        layer = {
            'layer_type': layer_type,
            'block_type': block_type,
            'param': param,
        }

        if layer_type in {'conv', 'transpose_conv', 'max_pool', 'upsample'}:

            if not isinstance(param, int):
                raise ValueError(
                    f"Kernel size must be an int, but received a "
                    f"value of type {type(param)}."
                )

            use_max_pool = True if layer_type == 'max_pool' else use_max_pool
            use_upsample = True if layer_type == 'upsample' else use_upsample
            num_convs += 1 if layer_type in {'conv', 'max_pool'} else 0

            if layer_type in {'upsample', 'transpose_conv'}:
                num_transpose += 1

            is_first_conv = num_convs == 1 and block_type == 'encoder'
            is_first_transpose = num_transpose == 1 and block_type == 'decoder'

            if (layer_type == 'conv' and is_first_conv) \
                    or (layer_type == 'transpose' and is_first_transpose):
                processed_block = [layer] + processed_block
                continue

        elif layer_type == 'dropout' or layer_type == 'leaky_relu':
            if not isinstance(param, float):
                raise ValueError(
                    f"Value for {layer_type} must be a float, but "
                    f"received a value of type {type(param)}."
                )
        else:
            assert layer_type in {'batch_norm', 'relu', 'sigmoid', 'tanh'}, \
                f"Unknown layer {layer_type} was passed in."

        processed_block.append(layer)

    num_layers = len(processed_block)
    if block_type == 'encoder':
        assert num_convs > 0 and num_layers > 0, \
            f"Must specify at least 1 convolutional layer."
    elif block_type == 'decoder':
        assert num_transpose > 0 and num_layers > 0, \
            f"Must specify at least 1 transpose convolution or upsample layer."

    processed_block = processed_block
    if block_type == 'encoder':
        down_sampling_stack = []
        stop_layer = 'max_pool' if use_max_pool else 'conv'
        while len(processed_block) > 0:
            layer = processed_block.pop()
            down_sampling_stack.append(layer)
            if layer['layer_type'] == stop_layer:
                layer['down'] = True
                if num_convs > 1:
                    break
        processed_block = {
            'conv_stack': processed_block,
            'downsampling_stack': down_sampling_stack[::-1]
        }
    elif block_type == 'decoder':
        up_sampling_stack = []
        stop_layer = 'upsample' if use_upsample else 'transpose_conv'
        while len(processed_block) > 0:
            layer = processed_block[0]
            if layer['layer_type'] == stop_layer:
                if stop_layer in {'transpose_conv', 'upsample'}:
                    layer['up'] = True
                elif stop_layer == 'conv':
                    break
                stop_layer = 'conv'
            up_sampling_stack.append(processed_block.pop(0))
        processed_block = {
            'conv_stack': processed_block,
            'upsampling_stack': up_sampling_stack
        }
    else:
        pass

    return processed_block


def build_layers(config_dict: dict):
    d1 = yaml_parser.load(Path('/Users/Kian/Desktop/auralflow/config/unet/unet_base.yaml'))
    d2 = yaml_parser.load(Path('/Users/Kian/Desktop/auralflow/config/data_config.yaml'))
    d = d1 | d2


    m = build_model(d)
    pprint(m)
    #
    x = torch.rand((1, 512, 173, 1))
    try:
        m(x)

        torchinfo.summary(m, x.size(), depth=8)
    except ValueError:
        raise
    return
#
# def parse_layers(config_dict: dict):
