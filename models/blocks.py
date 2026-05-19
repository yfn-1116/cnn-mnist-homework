import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
