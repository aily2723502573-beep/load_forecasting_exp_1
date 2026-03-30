import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """
    A simple fully-connected N-BEATS block that outputs theta = [backcast, forecast].
    Added:
      - configurable num_layers
      - dropout between hidden layers
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_dim = input_size
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_size
        layers += [nn.Linear(in_dim, theta_size)]
        self.fc = nn.Sequential(*layers)

        self.theta_size = theta_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class NBEATS(nn.Module):
    """
    Minimal N-BEATS-style model:
      - iteratively refines residual via backcast
      - sums block forecasts

    Expected input:
      - if x is (B, T, F): flatten to (B, T*F)
      - if x is (B, D): use as-is
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_blocks: int = 2,        # changed default: 2 (more comparable capacity)
        hidden_size: int = 128,     # changed default: 128
        num_layers: int = 4,        # common N-BEATS choice
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        theta_size = input_size + output_size
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)  # (B, T*F)

        residual = x
        forecast = torch.zeros(x.size(0), self.output_size, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            theta = block(residual)
            backcast = theta[:, : self.input_size]
            fcast = theta[:, self.input_size :]

            residual = residual - backcast
            forecast = forecast + fcast

        return forecast