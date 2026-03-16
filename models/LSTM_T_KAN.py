import torch
import torch.nn as nn

from .KAN import KAN


class LSTM_T_KAN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_lstm_layers=2,
        kan_layers=None,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        if kan_layers is None:
            kan_layers = [hidden_size, hidden_size, 32, 1]

        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.kan = KAN(layers_hidden=kan_layers)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_directions * self.lstm.num_layers, batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_directions * self.lstm.num_layers, batch_size, self.hidden_size
        ).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_out = out[:, -1, :]
        output = self.kan(last_out)
        return output
