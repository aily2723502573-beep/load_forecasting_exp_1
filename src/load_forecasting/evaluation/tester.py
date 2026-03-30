import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from ..models.factory import FLATTEN_INPUT_MODELS, TREE_MODELS, build_model

PERSISTENCE_MODEL = "Persistence"


class ModelTester:
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        output_dim: int,
        building_name: str,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints",
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.input_dim = input_dim
        self.window_size = window_size
        self.output_dim = output_dim
        self.building_name = building_name
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.model_config = model_config or {}

    def _test_tree_model(
        self, X: np.ndarray, y: np.ndarray, horizon_steps: int, model_type: str, scaler_y: MinMaxScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        import joblib

        model_path = os.path.join(self.checkpoint_dir, f"{self.building_name}_{model_type}_len{horizon_steps}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        X_flat = np.asarray(X).reshape(X.shape[0], -1)
        y_pred = model.predict(X_flat).astype(np.float32)
        y_true = np.asarray(y).astype(np.float32)

        original_shape = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, self.output_dim)
        y_true_flat = y_true.reshape(-1, self.output_dim)
        y_pred_inv = scaler_y.inverse_transform(y_pred_flat)
        y_true_inv = scaler_y.inverse_transform(y_true_flat)
        return y_pred_inv.reshape(original_shape), y_true_inv.reshape(original_shape)

    def _build_model(self, model_type: str, horizon_steps: int):
        cfg = self.model_config.get(model_type, {})
        return build_model(
            model_type=model_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            window_size=self.window_size,
            horizon_steps=horizon_steps,
            model_config=cfg,
        )

    def _test_persistence(
        self, X: np.ndarray, y: np.ndarray, horizon_steps: int, scaler_y: MinMaxScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Persistence: ŷ_{t+h} = y_t (repeat last observed value for all horizons)."""
        if self.output_dim != 1:
            raise ValueError(
                f"Persistence baseline supports only univariate targets (output_dim == 1); got {self.output_dim}."
            )
        X_arr = np.asarray(X)
        last_val = X_arr[:, -1, -1]  # [N]
        y_pred = np.tile(last_val[:, np.newaxis], (1, horizon_steps)).astype(np.float32)
        y_true = np.asarray(y).astype(np.float32)

        original_shape = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, self.output_dim)
        y_true_flat = y_true.reshape(-1, self.output_dim)
        y_pred_inv = scaler_y.inverse_transform(y_pred_flat)
        y_true_inv = scaler_y.inverse_transform(y_true_flat)
        return y_pred_inv.reshape(original_shape), y_true_inv.reshape(original_shape)

    def test(
        self, X: np.ndarray, y: np.ndarray, horizon_steps: int, model_type: str, scaler_y: MinMaxScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        if model_type == PERSISTENCE_MODEL:
            return self._test_persistence(X, y, horizon_steps, scaler_y)
        if model_type in TREE_MODELS:
            return self._test_tree_model(X, y, horizon_steps, model_type, scaler_y)

        model_path = os.path.join(self.checkpoint_dir, f"{self.building_name}_{model_type}_len{horizon_steps}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        net = self._build_model(model_type, horizon_steps)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and any(k.startswith("func.") for k in state.keys()):
            remapped = {}
            for k, v in state.items():
                if k.startswith("func."):
                    remapped["fc." + k[len("func.") :]] = v
                else:
                    remapped[k] = v
            state = remapped
        net.load_state_dict(state, strict=True)
        net = net.to(self.device)
        net.eval()

        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_true = np.asarray(y, dtype=np.float32)

        with torch.no_grad():
            if model_type in FLATTEN_INPUT_MODELS:
                X_reshaped = X_tensor.reshape(X_tensor.shape[0], -1)
                y_pred_tensor = net(X_reshaped)
            else:
                y_pred_tensor = net(X_tensor)

        y_pred = y_pred_tensor.cpu().numpy()

        original_shape = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, self.output_dim)
        y_true_flat = y_true.reshape(-1, self.output_dim)

        y_pred_inv = scaler_y.inverse_transform(y_pred_flat)
        y_true_inv = scaler_y.inverse_transform(y_true_flat)

        y_pred_inv = y_pred_inv.reshape(original_shape)
        y_true_inv = y_true_inv.reshape(original_shape)

        return y_pred_inv, y_true_inv
