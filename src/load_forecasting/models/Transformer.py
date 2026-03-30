import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.drop(x)


class Transformer(nn.Module):
    """
    Transformer Encoder baseline for time series.
    - input x: (B, T, F)
    - output: (B, output_size)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,            # changed default: 64
        nhead: int = 4,               # changed default: 4 (must divide d_model)
        num_layers: int = 2,          # changed default: 2
        dim_feedforward: int = 128,   # changed default: 128
        output_size: int = 1,
        dropout: float = 0.2,
        pooling: str = "mean",        # "mean" or "last"
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert pooling in ("mean", "last")

        self.pooling = pooling
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # usually helps stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)     # (B, T, d_model)
        x = self.pos_encoder(x)    # (B, T, d_model)
        x = self.transformer(x)    # (B, T, d_model)

        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        return self.fc(x)