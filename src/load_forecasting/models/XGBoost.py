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
    """
    Create an XGBRegressor with strong, stable defaults.

    Early stopping:
      - Works only if you call fit(..., eval_set=[(X_val, y_val)], early_stopping_rounds=...)
      - Example:
          model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=10)
    """
    xgb = _get_xgboost()
    cfg = config or {}

    seed = cfg.get("seed", cfg.get("random_state", cfg.get("random_seed", 42)))

    return xgb.XGBRegressor(
        # --- capacity / training ---
        n_estimators=cfg.get("n_estimators", 5000),
        learning_rate=cfg.get("learning_rate", 0.03),
        max_depth=cfg.get("max_depth", 6),
        min_child_weight=cfg.get("min_child_weight", 1),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),

        # --- regularization ---
        reg_alpha=cfg.get("reg_alpha", 0.0),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        gamma=cfg.get("gamma", 0.0),

        # --- objective/metric ---
        objective=cfg.get("objective", "reg:squarederror"),
        eval_metric=cfg.get("eval_metric", "rmse"),

        # --- runtime ---
        tree_method=cfg.get("tree_method", "hist"),   # CPU: "hist"; GPU: "gpu_hist"
        n_jobs=cfg.get("n_jobs", -1),

        # --- reproducibility ---
        random_state=seed,
    )