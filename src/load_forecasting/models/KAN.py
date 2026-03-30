import math

import torch
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = torch.randn_like(self.spline_weight) * 0.1 / self.grid_size
            self.spline_weight.data.copy_(noise)
            torch.nn.init.constant_(self.spline_scaler, self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"KANLinear.b_splines expected input of shape (batch_size, {self.in_features}), "
                f"but got tensor with shape {tuple(x.shape)}."
            )
        grid = self.grid
        x = x.unsqueeze(-1)
        x = x.clamp(grid[:, :1], grid[:, -1:])

        bases = (x >= grid[:, :-1]) & (x < grid[:, 1:])
        at_right = (x >= grid[:, -1:]).squeeze(-1)
        if at_right.any():
            bases = bases.clone()
            bases[:, :, -1] = bases[:, :, -1] | at_right
        bases = bases.to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"KANLinear.forward expected input with last dimension {self.in_features}, "
                f"but got tensor with shape {tuple(x.shape)}."
            )
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_bases = self.b_splines(x)
        spline_output = torch.einsum(
            "bik,oik->bo", spline_bases, self.spline_weight * self.spline_scaler.unsqueeze(-1)
        )
        return base_output + spline_output

    def regularization_loss(self):
        return self.spline_weight.abs().mean()


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self):
        return sum(layer.regularization_loss() for layer in self.layers)
