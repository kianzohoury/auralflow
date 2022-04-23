from models.base_unet import BaseUNet

REQUIRED_MODEL_KEYS = {
    'init_hidden',
    'num_fft',
    'hop_length',
    'num_channels',
    'targets',
    'window_size',
    'bottleneck_layers',
    'bottleneck_type',
    'max_layers',
    'encoder',
    'decoder',
    'num_dropouts',
    'skip_connections',
    'mask_activation',
    'input_norm',
    'output_norm',
}

BASE_MODELS = {
    "u-net": BaseUNet
}

