import torch

# from models.base import UNetSpec, UNetVAESpec
from models.static_models import SpectrogramNetSimple

if __name__ == "__main__":
    # autoencoder = AutoEncoder2d(
    #     num_targets=3,
    #     num_bins=512,
    #     num_samples=173,
    #     num_channels=1,
    #     max_depth=6,
    #     hidden_size=16,
    #     kernel_size=(3, 5, 3, 5),
    #     same_padding=True,
    #     block_size=3,
    #     downsampler='conv',
    #     upsampler='transpose',
    #     batch_norm=True,
    #     activation='leaky_relu',
    #     use_skip=True,
    #     dropout_p=0.5,
    #     leak_constant=0.2
    # )

    # model = TFMaskUNet(
    #     num_fft_bins=1023,
    #     num_samples=315,
    #     num_channels=2,
    #     kernel_size=(3, 2, 5, 3),
    #     normalize_input=True,
    #     mask_activation_fn="sigmoid",
    # )

    # print(model.autoencoder.kernel_size)
    # summary(model, input_size=(1, 1, 3 * 44100))
    # x, y = torch.rand((1, 2, 44100 * 2)), torch.rand((1, 2, 44100 * 2, 4))
    # print(_get_transpose_padding(15, 8, 31, 18, 2, 5))
    # xx, yy = model.process_input(x, y)
    # print(xx.shape, yy.shape)
    # print(model.num_fft, model.num_bins, model.window_size)
    # print(model(torch.rand((1, 2, 512, 315))).shape)
    # print(model(torch.rand((1, 1, 3 * 44100))).size())
    # vae(torch.rand((1, 1, 512, 173)))
    #
    # print(ConvBlock(2, 128, 256, 2, 1, dropout_p=.8))
    model = SpectrogramNetSimple(512, 173)
    print(model(torch.rand((1, 1, 512, 173))))
