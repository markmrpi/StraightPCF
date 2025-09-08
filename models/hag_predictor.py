import torch
import torch.nn as nn


class HAGPredictor(nn.Module):
    """Simple multilayer perceptron to regress HAG from XYZ coordinates."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz)
