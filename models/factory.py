from typing import Any, Dict, Union

import torch.nn as nn

from .CatBoost import create_catboost
from .GRU import GRU
from .KAN import KAN
from .LSTM import LSTM
from .LSTM_MLP import LSTM_MLP
from .LSTM_T_KAN import LSTM_T_KAN
from .MLP import MLP
from .NBEATS import NBEATS
from .TCN import TCN
from .Transformer import Transformer
from .XGBoost import create_xgboost


def build_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    window_size: int,
    horizon_steps: int,
    model_config: Dict[str, Any] | None = None,
) -> Union[nn.Module, Any]:
    """
    Build a forecasting model from config.

    Args:
        model_type: One of MLP, GRU, LSTM, LSTM_MLP, KAN, LSTM_T_KAN, TCN, NBEATS, Transformer, CatBoost, XGBoost.
        input_dim: Input feature dimension.
        output_dim: Output target dimension.
        window_size: Look-back window length.
        horizon_steps: Forecast horizon length (output steps).
        model_config: Per-model hyperparameters from config. If None, uses defaults.

    Returns:
        The constructed model.
    """
    cfg = model_config or {}

    output_size = horizon_steps * output_dim
    if model_type == "MLP":
        return MLP(
            input_size=input_dim * window_size,
            hidden_size1=cfg.get("hidden_size1", 64),
            hidden_size2=cfg.get("hidden_size2", 32),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
        )
    if model_type == "GRU":
        return GRU(
            input_size=input_dim,
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 2),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
        )
    if model_type == "LSTM":
        return LSTM(
            input_size=input_dim,
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 2),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
        )
    if model_type == "KAN":
        return KAN(
            layers_hidden=[input_dim * window_size, horizon_steps * output_dim],
            grid_size=cfg.get("grid_size", 5),
            spline_order=cfg.get("spline_order", 3),
        )
    if model_type == "LSTM_T_KAN":
        hidden_size = cfg.get("hidden_size", 128)
        bidirectional = cfg.get("bidirectional", False)
        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        kan_layers = list(cfg.get("kan_layers", [128, 64, 32]))
        kan_layers = [lstm_output_dim] + kan_layers[1:] + [output_size]
        return LSTM_T_KAN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_lstm_layers=cfg.get("num_lstm_layers", 2),
            kan_layers=kan_layers,
            dropout=cfg.get("dropout", 0.2),
            bidirectional=bidirectional,
        )
    if model_type == "LSTM_MLP":
        hidden_size = cfg.get("hidden_size", 128)
        bidirectional = cfg.get("bidirectional", False)
        return LSTM_MLP(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_lstm_layers=cfg.get("num_lstm_layers", 2),
            mlp_hidden1=cfg.get("mlp_hidden1", 64),
            mlp_hidden2=cfg.get("mlp_hidden2", 32),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
            bidirectional=bidirectional,
        )
    if model_type == "TCN":
        num_channels = cfg.get("num_channels", [32, 64])
        return TCN(
            input_size=input_dim,
            num_channels=num_channels,
            kernel_size=cfg.get("kernel_size", 3),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
        )
    if model_type == "NBEATS":
        input_size = input_dim * window_size
        return NBEATS(
            input_size=input_size,
            output_size=output_size,
            num_blocks=cfg.get("num_blocks", 4),
            hidden_size=cfg.get("hidden_size", 64),
            dropout=cfg.get("dropout", 0.2),
        )
    if model_type == "Transformer":
        return Transformer(
            input_size=input_dim,
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 128),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
        )
    if model_type == "CatBoost":
        base = create_catboost(cfg)
        if horizon_steps > 1:
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(base, n_jobs=-1)
        return base
    if model_type == "XGBoost":
        base = create_xgboost(cfg)
        if horizon_steps > 1:
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(base, n_jobs=-1)
        return base

    raise ValueError(f"Unknown model type: {model_type}")


# Models that expect flattened input [B, T*F]
FLATTEN_INPUT_MODELS = ("KAN", "MLP", "NBEATS", "CatBoost", "XGBoost")

# Tree models: use fit/predict, save with joblib
TREE_MODELS = ("CatBoost", "XGBoost")
