import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=None, num_classes=10, dropout=0.0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
