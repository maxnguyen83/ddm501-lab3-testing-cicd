"""
Unit tests for configuration and the model singleton helpers.

Run tests:
    pytest tests/unit/test_utils.py -v
"""

import importlib
import re
from pathlib import Path

import pytest

import app
import app.config as config
import app.model as model_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reload_config(monkeypatch):
    """
    Reload app.config after changing environment variables.

    Teardown undoes the env changes first and reloads again, so later tests
    see the default configuration.
    """

    def _reload(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)


class TestConfig:
    """Tests for app.config."""

    def test_default_model_path_points_into_models_dir(self, reload_config, monkeypatch):
        """Without MODEL_PATH set, the model is read from <project>/models."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        cfg = reload_config()
        assert Path(cfg.MODEL_PATH) == PROJECT_ROOT / "models" / "svd_model.pkl"

    def test_env_overrides(self, reload_config):
        """Environment variables override the defaults."""
        cfg = reload_config(MODEL_PATH="/tmp/other.pkl", MODEL_VERSION="2.0.0", PORT="9000")
        assert cfg.MODEL_PATH == "/tmp/other.pkl"
        assert cfg.MODEL_VERSION == "2.0.0"
        assert cfg.PORT == 9000

    @pytest.mark.parametrize(
        "raw, expected", [("true", True), ("TRUE", True), ("false", False), ("1", False)]
    )
    def test_debug_flag_parsing(self, reload_config, raw, expected):
        """Only the string 'true' (any case) turns DEBUG on."""
        assert reload_config(DEBUG=raw).DEBUG is expected

    def test_invalid_port_fails_fast(self, reload_config):
        """A non-numeric PORT is a configuration error at import time."""
        with pytest.raises(ValueError):
            reload_config(PORT="not-a-number")

    def test_rating_bounds(self):
        """The rating scale matches MovieLens (1 to 5 stars)."""
        assert (config.MIN_RATING, config.MAX_RATING) == (1.0, 5.0)

    def test_versions_are_consistent(self):
        """Package, API and pyproject versions must not drift apart."""
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
        assert match is not None
        assert app.__version__ == config.API_VERSION == match.group(1)


class TestModelSingleton:
    """Tests for get_model() / reset_model()."""

    @pytest.fixture(autouse=True)
    def _isolate_singleton(self, monkeypatch):
        """Replace the real loader so these tests do not need a model file."""
        created = []

        def fake_model():
            created.append(object())
            return created[-1]

        model_module.reset_model()
        monkeypatch.setattr(model_module, "MovieRatingModel", fake_model)
        yield created
        model_module.reset_model()

    def test_get_model_returns_same_instance(self, _isolate_singleton):
        """The model is loaded once and then reused."""
        first = model_module.get_model()
        second = model_module.get_model()
        assert first is second
        assert len(_isolate_singleton) == 1

    def test_reset_model_forces_reload(self, _isolate_singleton):
        """After reset_model() the next call creates a new instance."""
        first = model_module.get_model()
        model_module.reset_model()
        second = model_module.get_model()
        assert first is not second
        assert len(_isolate_singleton) == 2


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
