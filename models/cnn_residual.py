import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))

        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity
        return self.relu(out)


class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3, use_batchnorm: bool = False):
        super().__init__()

        stem = [
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            stem.append(nn.BatchNorm2d(32))
        stem.append(nn.ReLU(inplace=True))
        stem.append(nn.MaxPool2d(2))

        stem.extend([
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_batchnorm),
        ])
        if use_batchnorm:
            stem.append(nn.BatchNorm2d(64))
        stem.append(nn.ReLU(inplace=True))
        stem.append(nn.MaxPool2d(2))

        self.stem = nn.Sequential(*stem)
        self.res_block = ResidualBlock(64, use_batchnorm=use_batchnorm)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_block(x)
        x = self.classifier(x)
        return x
