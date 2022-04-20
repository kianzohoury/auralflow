import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.input_norm = nn.BatchNorm2d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, 2, 2), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, 2, 2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 5, 2, 2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 512, 5, 2, 2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.deconv1 = nn.ConvTranspose2d(512, 256, 5, 2, 2)
        self.a1 = nn.Sequential(nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.ReLU())
        self.deconv2 = nn.ConvTranspose2d(512, 128, 5, 2, 2)
        self.a2 = nn.Sequential(nn.BatchNorm2d(128), nn.Dropout2d(0.5), nn.ReLU())
        self.deconv3 = nn.ConvTranspose2d(256, 64, 5, 2, 2)
        self.a3 = nn.Sequential(nn.BatchNorm2d(64), nn.Dropout2d(0.5), nn.ReLU())
        self.deconv4 = nn.ConvTranspose2d(128, 32, 5, 2, 2)
        self.a4 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU())
        self.deconv5 = nn.ConvTranspose2d(64, 16, 5, 2, 2)
        self.a5 = nn.Sequential(nn.BatchNorm2d(16), nn.ReLU())
        self.deconv6 = nn.ConvTranspose2d(32, 1, 5, 2, 2)

        self.final_conv = nn.Conv2d(1, 1, 1, 1, padding='same')
        self.output_norm = nn.BatchNorm2d(512)
        self.a6 = nn.Sigmoid()

    def forward(self, x):

        x = self.input_norm(x)

        x = x.permute(0, 3, 1, 2)
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)

        d1 = self.a1(self.deconv1(e6, output_size=e5.size()))
        d2 = self.a2(self.deconv2(torch.cat([d1, e5], dim=1), output_size=e4.size()))
        d3 = self.a3(self.deconv3(torch.cat([d2, e4], dim=1), output_size=e3.size()))
        d4 = self.a4(self.deconv4(torch.cat([d3, e3], dim=1), output_size=e2.size()))
        d5 = self.a5(self.deconv5(torch.cat([d4, e2], dim=1), output_size=e1.size()))
        d6 = self.a6(self.deconv6(torch.cat([d5, e1], dim=1), output_size=x.size()))

        d6 = self.final_conv(d6)
        out = d6.permute(0, 2, 3, 1)
        out = self.output_norm(out)
        out = self.a6(out)

        output = {
            'mask': out,
            'estimate': None
        }

        return output

