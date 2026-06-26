"""PyTorch model definition for Titanic survival classification."""

import torch
import torch.nn as nn


class TitanicMLP(nn.Module):
    """Feed-forward binary classifier over engineered Titanic features."""

    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
