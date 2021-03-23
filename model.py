from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'Generator',
    'Discriminator'
]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding..., )

            # Deconv -> BN -> ReLU -> Conv
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=True),
            # state size: (ngf * 8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(True),
            nn.Conv2d(ngf * 6, ngf * 6, 3, 1, 1, bias=True),
            # state size: (ngf * 6) x 8 x 8

            nn.ConvTranspose2d(ngf * 6, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=True),
            # state size: (ngf * 4) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=True),
            # state size: (ngf * 2) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            # state size: ngf x 64 x 64

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: ngf x 128 x 128

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: ngf x 256 x 256

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: nc x 512 x 512
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512

            # Conv2d(3x3, s1) -> Conv2d(3x3, s2) -> BN -> LeakyReLU
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256

            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 128 x 128

            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 64 x 64

            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 32 x 32

            nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 16, ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 16 x 16

            nn.Conv2d(ndf * 16, ndf * 16, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 16, ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8

            nn.Conv2d(ndf * 16, ndf * 32, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 32, ndf * 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 4 x 4

            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(ndf * 32, 1, 3, 1, 1, bias=False),
            # nn.Linear(ndf * 32, 1),
            # nn.Sigmoid()
        )
        self.linear = nn.Linear(ndf * 32, 1)
        self.apply(weights_init)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    netG = Generator(10, 64, 3, 1)
    print(netG)
    inputs = torch.randn([1, 10, 1, 1])
    print(netG(inputs).shape)

    netD = Discriminator(3, 64)
    print(netD)
    inputs2 = torch.randn([1, 3, 512, 512])
    print(netD(inputs2).shape)
