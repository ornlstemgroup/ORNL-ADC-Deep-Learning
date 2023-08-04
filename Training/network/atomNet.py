
import torch
import torch.nn as nn

import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class Block(nn.Module):
    def __init__(self, channel_in, channel_out, dim, kernel=3):
        super(Block, self).__init__()
        self.feature = nn.Sequential(
            conv(channel_in, dim, kernel),
            nn.BatchNorm2d(dim, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            conv(dim, channel_out, kernel))

    def forward(self, x):
        x = self.feature(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.feature = nn.Sequential(
            conv(dim, dim, 3),
            nn.BatchNorm2d(dim, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            conv(dim, dim, 3))

    def forward(self, x):
        x = self.feature(x)+x
        return x

class AtomNet(nn.Module):
    def __init__(self):
        super(AtomNet, self).__init__()
        self.dim = 256

        self.conv1 = nn.Conv2d(1, self.dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = nn.Sequential(ResidualBlock(self.dim),
                                    ResidualBlock(self.dim),
                                    ResidualBlock(self.dim),
                                    ResidualBlock(self.dim))
        self.block3 = Block(self.dim, 1, self.dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.block2(x)
        x3 = self.block3(x2)
        x4 = self.act(x3)
        return x4



if __name__ == "__main__":
    model = AtomNet()
    images1 = torch.zeros([2, 1, 256, 256])
    output = model(images1)
    print(output.shape)