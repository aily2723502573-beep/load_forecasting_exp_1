import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    1D causal convolution: output[t] only depends on x[:t].
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    A standard TCN residual block:
      causal conv -> relu -> dropout -> causal conv -> relu -> dropout + residual
    Uses dilation to enlarge receptive field.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.drop(out)

        return out + res


class TCN(nn.Module):
    """
    TCN baseline:
      input x: (B, T, F)
      output y: (B, output_size)

    Notes:
      - Uses dilations 1,2,4,... to cover long-range dependencies with few layers.
      - Dropout applied inside residual blocks (common practice).
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list,
        kernel_size: int = 3,
        output_size: int = 1,
        dropout: float = 0.3,   # align with your main model's regularization strength
    ):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2**i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.tcn(x)            # (B, C, T)
        last = x[:, :, -1]         # (B, C)
        return self.fc(last)       # (B, output_size)