import torch
import torch.nn as nn


class LSTM_MLP(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_lstm_layers: int = 2,
        mlp_hidden1: int = 64,
        mlp_hidden2: int = 32,
        output_size: int = 1,
        dropout: float = 0.1,
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
            dropout=dropout if num_lstm_layers > 1 else 0,
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
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_directions * self.lstm.num_layers, batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_directions * self.lstm.num_layers, batch_size, self.hidden_size
        ).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_out = out[:, -1, :]
        return self.mlp(last_out)
