from models.base_unet import BaseUNet

REQUIRED_MODEL_KEYS = {
    'init_features',
    'num_fft',
    'hop_length'
    'num_channels',
    'targets',
    'window_size'
    'bottleneck_layers',
    'bottleneck_type',
    'max_layers',
    'encoder_block_layers',
    'decoder_block_layers',
    'encoder_kernel_size',
    'decoder_kernel_size',
    'decoder_dropout',
    'num_dropouts',
    'skip_connections',
    'mask_activation',
    'encoder_down',
    'decoder_up',
    'encoder_leak',
    'input_norm',
    'output_norm'
}

BASE_MODELS = {
    "unet": BaseUNet
}

