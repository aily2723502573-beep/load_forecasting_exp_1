from typing import Any, Dict

_catboost = None


def _get_catboost():
    global _catboost
    if _catboost is None:
        try:
            import catboost as cb
            _catboost = cb
        except ImportError:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
    return _catboost


def create_catboost(config: Dict[str, Any] | None = None):
    cb = _get_catboost()
    cfg = config or {}

    seed = cfg.get("seed", cfg.get("random_seed", 42))

    return cb.CatBoostRegressor(
        # --- capacity / training (defaults are "moderate & fast") ---
        iterations=cfg.get("iterations", 200),          # default smaller (override in config)
        learning_rate=cfg.get("learning_rate", 0.05),  # slightly larger for fewer iters
        depth=cfg.get("depth", 4),

        # --- objective/metric ---
        loss_function=cfg.get("loss_function", "RMSE"),
        eval_metric=cfg.get("eval_metric", "RMSE"),

        # --- regularization / sampling ---
        l2_leaf_reg=cfg.get("l2_leaf_reg", 20.0),
        bagging_temperature=cfg.get("bagging_temperature", 5.0),
        random_strength=cfg.get("random_strength", 10.0),

        # feature (column) sampling: <1.0 makes it weaker + faster sometimes
        rsm=cfg.get("rsm", 0.7),

        # --- early stopping (patience-like) ---
        od_type=cfg.get("od_type", "Iter"),
        od_wait=cfg.get("od_wait", 10),

        # --- runtime / threading ---
        thread_count=cfg.get("thread_count", 4),  # IMPORTANT for speed / avoiding CPU oversubscription

        # --- logging / files ---
        verbose=cfg.get("verbose", 0),
        allow_writing_files=False,

        # --- reproducibility ---
        random_seed=seed,

        # --- optional GPU ---
        task_type=cfg.get("task_type", "CPU"),  # set "GPU" if available
        devices=cfg.get("devices", None),
    )