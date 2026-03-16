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
    return cb.CatBoostRegressor(
        iterations=cfg.get("iterations", 200),
        learning_rate=cfg.get("learning_rate", 0.1),
        depth=cfg.get("depth", 4),
        verbose=0,
        allow_writing_files=False,
        random_seed=cfg.get("random_seed", 42),
    )
