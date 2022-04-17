import config.utils
import sys
import torch.nn as nn
from pathlib import Path
import ruamel.yaml
from yaml import YAMLError
from config.constants import REQUIRED_MODEL_KEYS, BASE_MODELS
import models
from pprint import pprint

yaml_parser = ruamel.yaml.YAML(typ='safe', pure=True)
session_logs_file = Path(__file__).parent / 'session_logs.yaml'


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
        FileNotFoundError: If the model folder does not exist.
        YAMLError: If a configuration file cannot be read.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No such directory {model_dir} exists.")
    else:
        try:
            config_mappings = [
                yaml_parser.load(file) for file in model_dir.glob("*.yaml")
            ]
            combined_mappings = {}
            for data in config_mappings:
                combined_mappings.update(compress_keys(data))
            return combined_mappings
        except YAMLError:
            raise YAMLError(f"Error: cannot read files from {str(model_dir)}.")


def build_model(config_dict: dict) -> nn.Module:
    """Implements a PyTorch model from the configuration files.

    Args:
        config_dict (dict): Training configuration dictionary.

    Returns:
        (nn.Module): Pytorch model.
    """
    base_model = config_dict['model']['base_model']
    build_config = {}
    for _, config_map in config_dict.items():
        for param_name, param_val in config_map.items():
            if param_name in REQUIRED_MODEL_KEYS:
                build_config[param_name] = param_val
    return BASE_MODELS[base_model](**build_config)







#
# init_features: int = 16,
# num_fft: int = 512,
# num_channels: int = 1,
#
#
# bottleneck_layers: int = 1,
# bottleneck_type: str = 'conv',
# max_layers: int = 6,
# encoder_block_layers: int = 1,
# decoder_block_layers: int = 1,
# encoder_kernel_size: int = 5,
# decoder_kernel_size: int = 5,
# decoder_dropout: float = 0.5,
# num_dropouts: int = 3,
# skip_connections: bool = True,
# mask_activation: str = 'sigmoid',
# encoder_down: Optional[str] = 'conv',
# decoder_up: Optional[str] = 'transposed',
# encoder_leak: Optional[float] = 0.2,
# target: Union[int, list] = 1,
# input_norm: bool = False,
# output_norm: bool = False
#
#
# a = {'data':
#          {'backend': 'soundfile',
#           'hop-length': 768,
#           'num-channels': True,
#           'num-fft': 1024,
#           'sample-length': 3,
#           'sample-rate': 44100,
#           'targets': ['vocals'],
#           'window-size': 1024},
#  'model':
#      {'base-model': 'Unet',
#        'bottleneck': {'layers': 1, 'type': 'conv'},
#        'decoder': {'block-layers': 1,
#                    'dropout': 0.5,
#                    'kernel-size': 5,
#                    'num-dropouts': 3,
#                    'skip': True,
#                    'up': 'transposed'},
#        'depth': 6,
#        'encoder': {'block-layers': 1,
#                    'down': 'conv',
#                    'kernel-size': 5,
#                    'leak': 0.2},
#        'init-features': 16,
#        'input-normalization': False,
#        'mask-activation': 'sigmoid',
#        'name': 'my_model',
#        'output-normalization': False},
#  'training':
#          {'batch-size': 8,
#           'cuda': True,
#           'epochs': 100,
#           'gpus': 1,
#           'loss': 'l1',
#           'lr': 0.001,
#           'max-iters': 1000,
#           'num-workers': 8,
#           'optimizer': 'adam',
#           'patience': 10,
#           'persistent-workers': True,
#           'pin-memory': True,
#           'tensorboard': False,
#           'val-split': 0.2}}



