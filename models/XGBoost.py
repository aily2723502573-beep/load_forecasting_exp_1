from typing import Any, Dict

_xgboost = None


def _get_xgboost():
    global _xgboost
    if _xgboost is None:
        try:
            import xgboost as xgb

            _xgboost = xgb
        except ImportError:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
    return _xgboost


def create_xgboost(config: Dict[str, Any] | None = None):
    xgb = _get_xgboost()
    cfg = config or {}
    return xgb.XGBRegressor(
        n_estimators=cfg.get("n_estimators", 200),
        learning_rate=cfg.get("learning_rate", 0.1),
        max_depth=cfg.get("max_depth", 4),
        random_state=cfg.get("random_state", cfg.get("random_seed", 42)),
    )
