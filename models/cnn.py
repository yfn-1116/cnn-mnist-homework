import torch
import torch.nn as nn

from models.blocks import ConvBlock


class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3, use_batchnorm=False):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, use_batchnorm=use_batchnorm),
            ConvBlock(32, 64, use_batchnorm=use_batchnorm),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
