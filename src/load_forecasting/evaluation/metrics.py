import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def create_alpha_weights(tau: int, short_h: int = 24, mid_h: int = 168) -> np.ndarray:
    """
    Create horizon weights for weighted MAPE.
    - tau: total horizon length (e.g., 720 for 30 days)
    - short_h: short-term threshold (1-24h: weight 2.0)
    - mid_h: mid-term threshold (25-168h: weight 1.0)
    - long-term (169+h): weight 0.5
    """
    alpha = np.ones(tau, dtype=np.float32)
    alpha[:short_h] = 2.0
    alpha[short_h:mid_h] = 1.0
    alpha[mid_h:] = 0.5
    return alpha


def compute_weighted_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_steps: int,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute weighted MAPE with horizon-specific weights.
    Short-term (1-24h) weighted 2.0, mid (25-168h) 1.0, long (169+h) 0.5.
    NaN values in y_true or y_pred are ignored (excluded from the weighted average),
    consistent with compute_metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1, horizon_steps)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1, horizon_steps)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return float("nan")
    alpha = create_alpha_weights(horizon_steps)
    rel_error = np.abs((y_pred - y_true) / (np.abs(y_true) + epsilon))
    weights = np.broadcast_to(alpha, y_true.shape).astype(np.float32)
    weights = np.where(valid_mask, weights, 0.0)
    rel_error = np.where(valid_mask, rel_error, 0.0)
    numerator = np.sum(rel_error * weights, axis=-1)
    denominator = np.sum(weights, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_mape_per_sample = np.where(denominator > 0, numerator / denominator, np.nan)
    return float(np.nanmean(weighted_mape_per_sample))


def compute_metrics(actual: np.ndarray, predicted: np.ndarray):
    """
    Compute MAE, RMSE, R², and MAPE, ignoring NaN values.

    Returns:
        Tuple of (mae, rmse, r2, mape).
    """
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]

    mae = mean_absolute_error(actual_filtered, predicted_filtered)
    rmse = float(np.sqrt(mean_squared_error(actual_filtered, predicted_filtered)))
    r2 = r2_score(actual_filtered, predicted_filtered)
    mape = mean_absolute_percentage_error(actual_filtered, predicted_filtered)
    return mae, rmse, r2, mape
