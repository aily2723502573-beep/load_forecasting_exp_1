import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Plain LSTM baseline.
    - Uses last time step hidden state for prediction.
    - Does not manually create (h0, c0); PyTorch defaults to zeros.
    - Dropout inside nn.LSTM only works when num_layers > 1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,     # changed default (align to your baseline setting)
        num_layers: int = 1,       # changed default: 1
        output_size: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(lstm_output_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)      # default h0/c0 are zeros on correct device/dtype
        last_out = out[:, -1, :]   # (batch, hidden_size*num_directions)
        return self.fc(last_out)   # (batch, output_size)