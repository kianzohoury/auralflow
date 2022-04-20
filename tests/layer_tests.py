import unittest
import torch
import torch.nn as nn

from models.layers import EncoderBlock


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
        names = ['relu', 'sigmoid', 'glu', 'tanh', None]
        functions = [nn.ReLU, nn.Sigmoid, nn.GLU, nn.Tanh, nn.Identity]
        for name, func in zip(names, functions):
            block = EncoderBlock(16, 32, activation_fn=name, leak=0)
            self.assertIsInstance(block.activation, func)
        block = EncoderBlock(16, 32, activation_fn='relu', leak=0.9)
        self.assertEqual(block.activation.negative_slope, 0.9)
        self.assertIsInstance(block.activation, nn.LeakyReLU)

    def test_padding(self):
        data = torch.rand((8, 16, 512, 128))
        block = EncoderBlock(16, 32, 5, stride=2, padding=2)
        self.assertEqual(block.padding, 2)
        self.assertEqual(block(data).size(), torch.Size((8, 32, 256, 64)))
        block = EncoderBlock(16, 32, 5, stride=1, padding='same')
        self.assertEqual(block(data).size(), torch.Size((8, 32, 512, 128)))
        self.assertIsNone(block.conv.bias)
        block = EncoderBlock(16, 32, 5, bias=True)
        self.assertIsNotNone(block.conv.bias)

    def test_batchnorm(self):
        block = EncoderBlock(16, 32, batch_norm=False)
        self.assertIsInstance(block.batchnorm, nn.Identity)
        block = EncoderBlock(16, 32)
        self.assertIsInstance(block.batchnorm, nn.BatchNorm2d)