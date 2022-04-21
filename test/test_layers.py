import unittest
import torch
import torch.nn as nn

from models.layers import EncoderBlock, DecoderBlock, StackedBlock


class EncoderTest(unittest.TestCase):
    def test_channels(self):
        block = EncoderBlock(16, out_channels=None)
        self.assertEqual(block.conv.in_channels, 16)
        self.assertEqual(block.conv.out_channels, 32)
        block = EncoderBlock(16, 32)
        self.assertEqual(block.conv.in_channels, 16)
        self.assertEqual(block.conv.out_channels, 32)
        data = torch.rand((8, 16, 512, 128))
        self.assertEqual(block(data).size()[1], 32)

    def test_kernels(self):
        block = EncoderBlock(16, 32, kernel_size=5)
        self.assertEqual(block.kernel_size, 5)
        block = EncoderBlock(16, 32, kernel_size=(3, 3))
        self.assertEqual(block.kernel_size, (3, 3))

    def test_activations(self):
        block = EncoderBlock(16, 32, activation_fn='relu')
        self.assertIsInstance(block.activation, nn.ReLU)
        block = EncoderBlock(16, 32, activation_fn='leaky_relu', leak=0.2)
        self.assertIsInstance(block.activation, nn.LeakyReLU)
        self.assertEqual(block.activation.negative_slope, 0.2)
        block = EncoderBlock(16, 32, activation_fn='sigmoid')
        self.assertIsInstance(block.activation, nn.Sigmoid)
        block = EncoderBlock(16, 32, activation_fn='tanh')
        self.assertIsInstance(block.activation, nn.Tanh)
        block = EncoderBlock(16, 32, activation_fn=None)
        self.assertIsInstance(block.activation, nn.Identity)

    def test_padding_stride(self):
        data = torch.rand((8, 16, 512, 128))
        block = EncoderBlock(16, 32, 5, stride=2, padding=2)
        self.assertEqual(block.padding, 2)
        self.assertEqual(block.stride, 2)
        self.assertEqual(block.conv.stride, (2, 2))
        self.assertEqual(block(data).size(), torch.Size((8, 32, 256, 64)))
        block = EncoderBlock(16, 32, 5, stride=1, padding='same')
        self.assertEqual(block(data).size(), torch.Size((8, 32, 512, 128)))

    def test_batchnorm(self):
        block = EncoderBlock(16, 32, batch_norm=False)
        self.assertIsInstance(block.batchnorm, nn.Identity)
        block = EncoderBlock(16, 32)
        self.assertIsInstance(block.batchnorm, nn.BatchNorm2d)
        self.assertFalse(block.bias)
        self.assertIsNone(block.conv.bias)
        block = EncoderBlock(16, 32, 5, bias=True)
        self.assertIsNotNone(block.conv.bias)


class DecoderTest(unittest.TestCase):
    def test_channels(self):
        block = DecoderBlock(16, out_channels=None)
        self.assertEqual(block.convT.in_channels, 16)
        self.assertEqual(block.convT.out_channels, 8)
        block = DecoderBlock(16, 8)
        self.assertEqual(block.convT.in_channels, 16)
        self.assertEqual(block.convT.out_channels, 8)
        data = torch.rand((2, 16, 512, 128))
        output_size = torch.Size((2, 8, 1024, 256))
        output = block(data, output_size=output_size)
        self.assertEqual(output.size(), output_size)

    def test_kernels(self):
        block = DecoderBlock(16, 8, kernel_size=5)
        self.assertEqual(block.kernel_size, 5)
        block = DecoderBlock(16, 8, kernel_size=(3, 3))
        self.assertEqual(block.kernel_size, (3, 3))
        block = DecoderBlock(16, 8, kernel_size=(3, 3))
        self.assertEqual(block.kernel_size, (3, 3))

    def test_activations(self):
        block = DecoderBlock(16, 32, activation_fn='relu')
        self.assertIsInstance(block.activation, nn.ReLU)
        block = DecoderBlock(16, 32, activation_fn='sigmoid')
        self.assertIsInstance(block.activation, nn.Sigmoid)
        block = DecoderBlock(16, 32, activation_fn='tanh')
        self.assertIsInstance(block.activation, nn.Tanh)
        block = DecoderBlock(16, 32, activation_fn=None)
        self.assertIsInstance(block.activation, nn.Identity)

    def test_padding_stride(self):
        data = torch.rand((8, 16, 512, 128))
        block = DecoderBlock(16, 8, 5, stride=2, padding=2)
        self.assertEqual(block.padding, 2)
        output = block(data, output_size=torch.Size((8, 8, 1024, 256)))
        self.assertEqual(output.size(), torch.Size((8, 8, 1024, 256)))

        block = DecoderBlock(16, 8, 5, stride=(2, 4), padding=2)
        conv_transpose = nn.ConvTranspose2d(16, 8, 5, stride=(2, 4), padding=2)
        self.assertEqual(block(data).shape, conv_transpose(data).shape)

    def test_dropout_batchnorm(self):
        block = DecoderBlock(16, 8, batch_norm=False)
        self.assertIsInstance(block.batchnorm, nn.Identity)
        block = DecoderBlock(16, 8)
        self.assertIsInstance(block.batchnorm, nn.BatchNorm2d)
        self.assertFalse(block.bias)
        self.assertIsNone(block.convT.bias)
        self.assertIsInstance(block.dropout, nn.Identity)
        self.assertEqual(block.dropout_p, 0)
        block = DecoderBlock(16, 8, dropout_p=0.2)
        self.assertIsInstance(block.dropout, nn.Dropout2d)
        self.assertTrue(block.dropout.p, 0.2)
        self.assertEqual(block.dropout_p, 0.2)


class StackedBlockTest(unittest.TestCase):
    def test_encoder_stacked(self):
        stacked_scheme = [
            {
                'in_channels': 16,
                'out_channels': 32,
                'kernel_size': 5,
                'stride': 1,
                'padding': 'same',
                'batch_norm': True,
                'activation_fn': 'leaky_relu',
                'leak': 0.2,
                'bias': False,
                'max_pool': True
            },
            {
                'in_channels': 32,
                'out_channels': 64,
                'kernel_size': 5,
                'stride': 2,
                'padding': 2,
                'batch_norm': False,
                'activation_fn': 'sigmoid',
                'bias': False,
                'max_pool': False
            },
            {
                'in_channels': 64,
                'out_channels': 128,
                'kernel_size': 5,
                'stride': 2,
                'padding': 2,
                'batch_norm': True,
                'activation_fn': 'relu',
                'bias': False,
                'max_pool': True
            }
        ]
        encoder = StackedBlock(stacked_scheme, block_type='encoder')
        for i, layer in enumerate(encoder.layers_stack):
            self.assertEqual(layer.in_channels,
                             stacked_scheme[i]['in_channels'])
            self.assertEqual(layer.out_channels,
                             stacked_scheme[i]['out_channels'])
            self.assertEqual(layer.kernel_size,
                             stacked_scheme[i]['kernel_size'])
            self.assertEqual(layer.stride,
                             stacked_scheme[i]['stride'])
            self.assertEqual(layer.padding,
                             stacked_scheme[i]['padding'])
            if stacked_scheme[i]['batch_norm']:
                self.assertIsInstance(layer.batchnorm, nn.BatchNorm2d)
            if stacked_scheme[i]['activation_fn']:
                self.assertIsInstance(layer.activation,
                                      nn.modules.activation.Module)
            if stacked_scheme[i]['max_pool']:
                self.assertIsInstance(layer.maxpool, nn.MaxPool2d)
            self.assertEqual(layer.bias, stacked_scheme[i]['bias'])

    def test_decoder_stacked(self):
        stacked_scheme = [
            {
                'in_channels': 64,
                'out_channels': 32,
                'kernel_size': 5,
                'stride': 2,
                'padding': 2,
                'batch_norm': True,
                'activation_fn': 'relu',
                'bias': False
            },
            {
                'in_channels': 32,
                'out_channels': 16,
                'kernel_size': 5,
                'stride': 2,
                'padding': 2,
                'batch_norm': False,
                'activation_fn': 'sigmoid',
                'bias': False,
            },
            {
                'in_channels': 16,
                'out_channels': 1,
                'kernel_size': 5,
                'stride': 2,
                'padding': 2,
                'batch_norm': True,
                'activation_fn': 'relu',
                'bias': False,
            }
        ]
        encoder = StackedBlock(stacked_scheme, block_type='decoder')
        for i, layer in enumerate(encoder.layers_stack):
            self.assertEqual(layer.in_channels,
                             stacked_scheme[i]['in_channels'])
            self.assertEqual(layer.out_channels,
                             stacked_scheme[i]['out_channels'])
            self.assertEqual(layer.kernel_size,
                             stacked_scheme[i]['kernel_size'])
            self.assertEqual(layer.stride,
                             stacked_scheme[i]['stride'])
            self.assertEqual(layer.padding,
                             stacked_scheme[i]['padding'])
            if stacked_scheme[i]['batch_norm']:
                self.assertIsInstance(layer.batchnorm, nn.BatchNorm2d)
            if stacked_scheme[i]['activation_fn']:
                self.assertIsInstance(layer.batchnorm,
                                      nn.modules.activation.Module)
            self.assertEqual(layer.bias, stacked_scheme[i]['bias'])
