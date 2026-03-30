# test_CatBoost.py

import sys
import types
import pytest

# Absolute import of the function under test
from load_forecasting.models.CatBoost import create_catboost, _catboost

@pytest.mark.usefixtures("reset_catboost_global")
class TestCreateCatboost:
    """
    Unit tests for the create_catboost function in load_forecasting.models.CatBoost.
    """

    @pytest.fixture(autouse=True)
    def reset_catboost_global(self):
        """
        Fixture to reset the _catboost global variable before each test.
        Ensures tests are isolated and do not interfere with each other's state.
        """
        import load_forecasting.models.CatBoost as catboost_mod
        catboost_mod._catboost = None

    # -------------------- Happy Path Tests --------------------

    @pytest.mark.happy_path
    def test_create_catboost_default_config(self):
        """
        Test that create_catboost returns a CatBoostRegressor with default parameters when no config is provided.
        """
        cb_regressor = create_catboost()
        # Check type
        assert cb_regressor.__class__.__name__ == "CatBoostRegressor"
        # Check default parameters
        assert cb_regressor.get_param("iterations") == 200
        assert cb_regressor.get_param("learning_rate") == 0.1
        assert cb_regressor.get_param("depth") == 4
        assert cb_regressor.get_param("random_seed") == 42
        assert cb_regressor.get_param("verbose") == 0
        assert cb_regressor.get_param("allow_writing_files") is False

    @pytest.mark.happy_path
    def test_create_catboost_custom_config(self):
        """
        Test that create_catboost returns a CatBoostRegressor with custom parameters when config is provided.
        """
        config = {
            "iterations": 10,
            "learning_rate": 0.5,
            "depth": 8,
            "random_seed": 123,
        }
        cb_regressor = create_catboost(config)
        assert cb_regressor.get_param("iterations") == 10
        assert cb_regressor.get_param("learning_rate") == 0.5
        assert cb_regressor.get_param("depth") == 8
        assert cb_regressor.get_param("random_seed") == 123
        assert cb_regressor.get_param("verbose") == 0
        assert cb_regressor.get_param("allow_writing_files") is False

    @pytest.mark.happy_path
    def test_create_catboost_partial_config(self):
        """
        Test that create_catboost uses defaults for missing config keys.
        """
        config = {
            "iterations": 50,
            "depth": 2,
        }
        cb_regressor = create_catboost(config)
        assert cb_regressor.get_param("iterations") == 50
        assert cb_regressor.get_param("depth") == 2
        # Defaults for missing keys
        assert cb_regressor.get_param("learning_rate") == 0.1
        assert cb_regressor.get_param("random_seed") == 42

    # -------------------- Edge Case Tests --------------------

    @pytest.mark.edge_case
    def test_create_catboost_empty_config(self):
        """
        Test that create_catboost handles an empty config dictionary and uses defaults.
        """
        cb_regressor = create_catboost({})
        assert cb_regressor.get_param("iterations") == 200
        assert cb_regressor.get_param("learning_rate") == 0.1
        assert cb_regressor.get_param("depth") == 4
        assert cb_regressor.get_param("random_seed") == 42

    @pytest.mark.edge_case
    def test_create_catboost_none_config(self):
        """
        Test that create_catboost handles None config and uses defaults.
        """
        cb_regressor = create_catboost(None)
        assert cb_regressor.get_param("iterations") == 200
        assert cb_regressor.get_param("learning_rate") == 0.1
        assert cb_regressor.get_param("depth") == 4
        assert cb_regressor.get_param("random_seed") == 42

    @pytest.mark.edge_case
    def test_create_catboost_invalid_config_keys(self):
        """
        Test that create_catboost ignores unknown config keys and uses defaults for known keys.
        """
        config = {
            "foo": "bar",
            "iterations": 5,
        }
        cb_regressor = create_catboost(config)
        assert cb_regressor.get_param("iterations") == 5
        assert cb_regressor.get_param("learning_rate") == 0.1
        assert cb_regressor.get_param("depth") == 4
        assert cb_regressor.get_param("random_seed") == 42

    @pytest.mark.edge_case
    def test_create_catboost_catboost_not_installed(monkeypatch):
        """
        Test that create_catboost raises ImportError with correct message if catboost is not installed.
        """
        # Remove catboost from sys.modules and patch import to raise ImportError
        monkeypatch.setitem(sys.modules, "catboost", None)
        import load_forecasting.models.CatBoost as catboost_mod

        def fake_import(name, *args, **kwargs):
            if name == "catboost":
                raise ImportError
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr(catboost_mod, "_catboost", None)
        monkeypatch.setattr(__builtins__, "__import__", fake_import)
        with pytest.raises(ImportError, match="CatBoost is required. Install with: pip install catboost"):
            catboost_mod.create_catboost()

    @pytest.mark.edge_case
    def test_create_catboost_catboost_global_cache(monkeypatch):
        """
        Test that _catboost global cache is used after first import.
        """
        import load_forecasting.models.CatBoost as catboost_mod
        # First call: should import catboost and set _catboost
        cb_regressor1 = catboost_mod.create_catboost()
        cb_mod1 = catboost_mod._catboost
        # Patch import to raise ImportError (should not be called)
        def fake_import(name, *args, **kwargs):
            if name == "catboost":
                raise ImportError
            return __import__(name, *args, **kwargs)
        monkeypatch.setattr(__builtins__, "__import__", fake_import)
        # Second call: should use cached _catboost, not raise ImportError
        cb_regressor2 = catboost_mod.create_catboost()
        cb_mod2 = catboost_mod._catboost
        assert cb_mod1 is cb_mod2
        assert cb_regressor2.__class__.__name__ == "CatBoostRegressor"

    @pytest.mark.edge_case
    def test_create_catboost_config_with_unusual_types(self):
        """
        Test that create_catboost handles config values of unusual types gracefully.
        """
        config = {
            "iterations": "100",  # string instead of int
            "learning_rate": [0.2],  # list instead of float
            "depth": None,  # None instead of int
            "random_seed": {},  # dict instead of int
        }
        # CatBoostRegressor will likely raise a ValueError or TypeError
        with pytest.raises((ValueError, TypeError)):
            create_catboost(config)