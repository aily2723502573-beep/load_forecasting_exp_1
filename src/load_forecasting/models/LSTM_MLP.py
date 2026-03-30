import torch
import torch.nn as nn


class LSTM_MLP(nn.Module):
    """
    LSTM encoder + MLP head.
    - Use last time step hidden state for prediction.
    - Do not manually create (h0, c0); PyTorch defaults to zeros.
    - Dropout inside nn.LSTM only works when num_lstm_layers > 1.
      We still apply dropout in the MLP head regardless.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_lstm_layers: int = 1,     # changed default: 1 (align with your LSTM_T_KAN)
        mlp_hidden1: int = 128,       # changed default: 128
        mlp_hidden2: int = 64,        # changed default: 64
        output_size: int = 1,
        dropout: float = 0.3,         # changed default: 0.3 (align with your setting)
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size*num_directions)
        last_out = out[:, -1, :]       # (batch, hidden_size*num_directions)
        return self.mlp(last_out)      # (batch, output_size)