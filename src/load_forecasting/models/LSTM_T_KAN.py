import torch
import torch.nn as nn
from .KAN import KAN


class LSTM_T_KAN(nn.Module):
    """
    V2 (stabler generalization): LSTM -> last_out -> base_head + tanh(alpha)*KAN(last_out_adapter)

    Changes vs your V2:
    1) alpha_init: make residual participate earlier (default 0.5)
    2) feat_dropout: light dropout on last_out to improve generalization
    3) alpha_init/feat_dropout are configurable
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.3,
        bidirectional: bool = False,
        kan_hidden=None,
        use_tanh_adapter: bool = True,
        kan_tanh_tau: float = 4.0,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        use_layernorm: bool = False,

        # NEW (V2+)
        alpha_init: float = 0.5,
        feat_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        H = hidden_size * self.num_directions

        self.use_tanh_adapter = use_tanh_adapter
        self.kan_tanh_tau = kan_tanh_tau
        self.use_layernorm = use_layernorm

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # NEW
        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout and feat_dropout > 0 else nn.Identity()

        # linear base head
        self.base_head = nn.Linear(H, output_size)

        # optional LN before KAN
        self.pre_kan_norm = nn.LayerNorm(H)

        # NEW: residual strength init (tanh(alpha) bounds it)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        if kan_hidden is None:
            kan_hidden = [128, 64]
        kan_layers = [H] + list(kan_hidden) + [output_size]
        self.kan = KAN(
            layers_hidden=kan_layers,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(
            self.num_directions * self.num_layers, x.size(0), self.hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.num_directions * self.num_layers, x.size(0), self.hidden_size, device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        last_out = out[:, -1, :]                 # [B,H]
        last_out = self.feat_drop(last_out)      # NEW

        base = self.base_head(last_out)          # [B,output_size]

        z = self.pre_kan_norm(last_out) if self.use_layernorm else last_out
        if self.use_tanh_adapter:
            z = torch.tanh(z / self.kan_tanh_tau)

        res = self.kan(z)                        # [B,output_size]
        return base + torch.tanh(self.alpha) * res

    def regularization_loss(self) -> torch.Tensor:
        return self.kan.regularization_loss()