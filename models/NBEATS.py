import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size),
        )
        self.theta_size = theta_size

    def forward(self, x):
        return self.fc(x)


class NBEATS(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_blocks: int = 4,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            theta_size = input_size + output_size
            self.blocks.append(NBeatsBlock(input_size, theta_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, input_size]
        forecast = torch.zeros(x.size(0), self.output_size, device=x.device)
        for block in self.blocks:
            theta = block(x)
            backcast = theta[:, : self.input_size]
            fcast = theta[:, self.input_size :]
            forecast = forecast + fcast
            x = x - backcast
        return self.dropout(forecast)
