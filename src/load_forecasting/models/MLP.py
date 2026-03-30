import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP baseline for windowed time-series input.

    Expected input:
      - if x is (B, T, F): flatten to (B, T*F)
      - if x is (B, D): use as-is
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int = 128,
        hidden_size2: int = 64,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow either (B, T, F) or (B, D)
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.net(x)