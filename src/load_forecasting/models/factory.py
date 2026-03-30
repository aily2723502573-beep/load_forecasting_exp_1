from __future__ import annotations

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
# ---------------- Tree models (sklearn-like) ----------------
from sklearn.multioutput import MultiOutputRegressor

def _as_list(x, default=None):
    if x is None:
        return [] if default is None else list(default)
    return list(x)


def _get_cfg(model_type: str, model_config: Dict[str, Any] | None) -> Dict[str, Any]:
    if model_config is None:
        return {}
    if isinstance(model_config, dict) and model_type in model_config and isinstance(model_config[model_type], dict):
        return model_config[model_type]
    return model_config


def build_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    window_size: int,
    horizon_steps: int,
    model_config: Dict[str, Any] | None = None,
) -> Union[nn.Module, Any]:
    cfg = _get_cfg(model_type, model_config)
    output_size = int(horizon_steps * output_dim)

    # ---------------- Deep learning models ----------------
    if model_type == "MLP":
        return MLP(
            input_size=int(input_dim * window_size),
            hidden_size1=cfg.get("hidden_size1", 128),
            hidden_size2=cfg.get("hidden_size2", 64),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.2),
        )

    if model_type == "GRU":
        return GRU(
            input_size=input_dim,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.3),
        )

    if model_type == "LSTM":
        return LSTM(
            input_size=input_dim,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),
        )

    if model_type == "LSTM_MLP":
        return LSTM_MLP(
            input_size=input_dim,
            hidden_size=cfg.get("hidden_size", 64),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            mlp_hidden1=cfg.get("mlp_hidden1", 128),
            mlp_hidden2=cfg.get("mlp_hidden2", 64),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),
        )

    if model_type == "TCN":
        return TCN(
            input_size=input_dim,
            num_channels=_as_list(cfg.get("num_channels", [64, 64, 64])),
            kernel_size=cfg.get("kernel_size", 3),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.3),
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
            pooling=cfg.get("pooling", "mean"),
        )

    if model_type == "NBEATS":
        # NBEATS in this repo expects flattened input
        return NBEATS(
            input_size=int(input_dim * window_size),
            output_size=output_size,
            num_blocks=cfg.get("num_blocks", 2),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.2),
        )

    if model_type == "KAN":
        kan_hidden = _as_list(cfg.get("kan_hidden", [128, 64]))
        layers_hidden = [int(input_dim * window_size)] + kan_hidden + [output_size]
        return KAN(
            layers_hidden=layers_hidden,
            grid_size=cfg.get("grid_size", 5),
            spline_order=cfg.get("spline_order", 3),
            scale_base=cfg.get("scale_base", 1.0),
            scale_spline=cfg.get("scale_spline", 1.0),
        )

    if model_type == "LSTM_T_KAN":
        cfg = _get_cfg("LSTM_T_KAN", model_config)

        return LSTM_T_KAN(
            input_size=input_dim,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_lstm_layers", 1),
            output_size=output_size,
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),

            kan_hidden=_as_list(cfg.get("kan_hidden", [128, 64])),
            use_layernorm=cfg.get("use_layernorm", False),

            use_tanh_adapter=cfg.get("use_tanh_adapter", True),
            kan_tanh_tau=cfg.get("kan_tanh_tau", 4.0),
            kan_grid_size=cfg.get("kan_grid_size", cfg.get("grid_size", 5)),
            kan_spline_order=cfg.get("kan_spline_order", cfg.get("spline_order", 3)),

            alpha_init=cfg.get("alpha_init", 0.5),
            feat_dropout=cfg.get("feat_dropout", 0.1),
        )

    if model_type == "CatBoost":
        base = create_catboost(cfg)
        if horizon_steps > 1:
            return MultiOutputRegressor(base, n_jobs=1)  
        return base

    if model_type == "XGBoost":
        base = create_xgboost(cfg)
        if horizon_steps > 1:
            return MultiOutputRegressor(base, n_jobs=1)  
        return base

FLATTEN_INPUT_MODELS = ("KAN", "MLP", "NBEATS", "CatBoost", "XGBoost")

TREE_MODELS = ("CatBoost", "XGBoost")