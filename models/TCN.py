import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: list,
        kernel_size: int = 3,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.dropout(x)
        return self.fc(x)
