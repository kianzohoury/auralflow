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
    'use_skip_connections',
    'soft_conv',
    'mask_activation',
    'normalize_input',
    'normalize_output',
}

LAYER_TYPES = {
    'head',
    'conv',
    'transpose_conv',
    'max_pool',
    'upsample',
    'downsample',
    'batch_norm',
    'dropout',
    'leaky_relu',
    'relu',
    'sigmoid',
    'tanh'
}

BASE_MODELS = {
    "u-net": BaseUNet
}

