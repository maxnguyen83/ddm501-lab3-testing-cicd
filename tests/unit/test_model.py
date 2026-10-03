"""
Unit tests for MovieRatingModel class.

Two groups of tests:
- Tests that use the real trained model (``trained_model`` fixture).
- Isolated tests that load a tiny stub algorithm from a temporary pickle, so the
  wrapper's own logic (clipping, rounding, ID checks, error paths) is tested
  without depending on what SVD happens to predict.

Run tests:
    pytest tests/unit/test_model.py -v
"""

import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple

import numpy as np
import pytest

from app.model import MovieRatingModel


class StubAlgo:
    """Picklable stand-in for a Surprise algorithm with a fixed estimate."""

    def __init__(self, est: float) -> None:
        self.est = est
        self.calls: List[Tuple[str, str]] = []

    def predict(self, uid: str, iid: str) -> SimpleNamespace:
        """Mimic surprise's Prediction object (only ``est`` is used)."""
        self.calls.append((uid, iid))
        return SimpleNamespace(est=self.est)


@pytest.fixture
def make_stub_model(tmp_path: Path) -> Callable[[float], MovieRatingModel]:
    """Factory: write a StubAlgo pickle to disk and load it through the wrapper."""

    def _make(est: float) -> MovieRatingModel:
        path = tmp_path / f"stub_{est}.pkl"
        with open(path, "wb") as f:
            pickle.dump(StubAlgo(est), f)
        return MovieRatingModel(model_path=str(path))

    return _make


class TestMovieRatingModel:
    """Unit tests for MovieRatingModel class."""

    # =========================================================================
    # Model Loading Tests
    # =========================================================================

    def test_model_loads_successfully(self, trained_model):
        """Test that model loads without errors."""
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model):
        """Test that model instance has the model attribute."""
        assert hasattr(trained_model, "model")
        assert trained_model.model is not None

    # =========================================================================
    # Prediction Return Type Tests
    # =========================================================================

    def test_predict_returns_float(self, trained_model):
        """Test that predict() returns a float value."""
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)

    # =========================================================================
    # Rating Range Tests
    # =========================================================================

    def test_predict_returns_value_in_valid_range(self, trained_model):
        """Test that predictions are within 1-5 range."""
        result = trained_model.predict("196", "242")
        assert 1.0 <= result <= 5.0

    def test_predict_multiple_pairs_all_in_range(self, trained_model, known_user_movie_pairs):
        """Test that all predictions are in valid range."""
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert 1.0 <= result <= 5.0, f"{pair} -> {result}"

    def test_predict_rounds_to_two_decimals(self, trained_model):
        """The API contract returns ratings with at most two decimals."""
        result = trained_model.predict("196", "242")
        assert result == round(result, 2)

    # =========================================================================
    # Batch Prediction Tests
    # =========================================================================

    def test_predict_batch_returns_list(self, trained_model):
        """Test that predict_batch() returns a list."""
        pairs = [("196", "242"), ("186", "302")]
        results = trained_model.predict_batch(pairs)
        assert isinstance(results, list)

    def test_predict_batch_returns_correct_length(self, trained_model):
        """Test that predict_batch() returns correct number of results."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        results = trained_model.predict_batch(pairs)
        assert len(results) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model):
        """Test that all batch predictions are in valid range."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377"), ("99999", "242")]
        results = trained_model.predict_batch(pairs)
        assert all(isinstance(r, float) and 1.0 <= r <= 5.0 for r in results)

    def test_predict_batch_empty_list_returns_empty_list(self, trained_model):
        """An empty batch is valid input and gives an empty result."""
        assert trained_model.predict_batch([]) == []

    # =========================================================================
    # is_loaded() Tests
    # =========================================================================

    def test_is_loaded_returns_bool(self, trained_model):
        """Test that is_loaded() returns a boolean."""
        result = trained_model.is_loaded()
        assert isinstance(result, bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model):
        """Test that is_loaded() returns True for loaded model."""
        assert trained_model.is_loaded() is True

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_predict_with_none_user_id(self, trained_model):
        """
        None must be rejected, not silently treated as an unknown user.

        Before the fix the wrapper passed None straight to Surprise, which
        returned the cold-start fallback rating with no error.
        """
        with pytest.raises(TypeError, match="user_id"):
            trained_model.predict(None, "242")

    def test_predict_with_empty_string(self, trained_model):
        """Empty or whitespace-only IDs raise ValueError."""
        with pytest.raises(ValueError, match="user_id"):
            trained_model.predict("", "242")
        with pytest.raises(ValueError, match="movie_id"):
            trained_model.predict("196", "   ")

    def test_predict_with_integer_ids_raises_type_error(self, trained_model):
        """Integer IDs would miss the string-keyed lookup, so they are rejected."""
        with pytest.raises(TypeError, match="movie_id"):
            trained_model.predict("196", 242)


class TestPredictionPostProcessing:
    """Isolated tests of the wrapper logic using a stub algorithm."""

    @pytest.mark.parametrize(
        "raw_est, expected",
        [
            (3.14159, 3.14),  # rounded to 2 decimals
            (7.3, 5.0),  # clipped to MAX_RATING
            (-2.0, 1.0),  # clipped to MIN_RATING
            (1.0, 1.0),  # lower boundary kept
            (5.0, 5.0),  # upper boundary kept
            (4.996, 5.0),  # rounds up onto the upper bound
        ],
    )
    def test_rounding_and_clipping(self, make_stub_model, raw_est, expected):
        """Raw estimates are rounded, then clipped into [1.0, 5.0]."""
        model = make_stub_model(raw_est)
        assert model.predict("1", "1") == expected

    def test_numpy_estimate_converted_to_builtin_float(self, make_stub_model):
        """A numpy scalar from the algorithm comes back as a plain float."""
        model = make_stub_model(np.float32(3.5))
        result = model.predict("1", "1")
        assert type(result) is float
        assert result == 3.5

    def test_ids_are_stripped_before_lookup(self, make_stub_model):
        """Surrounding whitespace is removed before the algorithm sees the IDs."""
        model = make_stub_model(3.0)
        model.predict("  196 ", "\t242\n")
        assert model.model.calls == [("196", "242")]

    def test_predict_batch_preserves_order(self, make_stub_model):
        """predict_batch calls the algorithm once per pair, in input order."""
        model = make_stub_model(3.0)
        pairs = [("1", "10"), ("2", "20"), ("3", "30")]
        assert model.predict_batch(pairs) == [3.0, 3.0, 3.0]
        assert model.model.calls == pairs

    def test_predict_raises_when_model_not_loaded(self, make_stub_model):
        """predict() refuses to run if the underlying model is missing."""
        model = make_stub_model(3.0)
        model.model = None
        assert model.is_loaded() is False
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict("1", "1")

    def test_predict_batch_raises_when_model_not_loaded(self, make_stub_model):
        """predict_batch() refuses to run if the underlying model is missing."""
        model = make_stub_model(3.0)
        model.model = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict_batch([("1", "1")])


class TestModelFileHandling:
    """Tests for model file handling."""

    def test_model_raises_error_for_missing_file(self):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="/nonexistent/path/model.pkl")

    def test_missing_file_is_logged(self, caplog):
        """The missing path is logged so operators can see what went wrong."""
        with caplog.at_level(logging.ERROR, logger="app.model"):
            with pytest.raises(FileNotFoundError):
                MovieRatingModel(model_path="/nonexistent/path/model.pkl")
        assert "/nonexistent/path/model.pkl" in caplog.text

    def test_corrupt_model_file_raises(self, tmp_path, caplog):
        """A file that is not a valid pickle raises instead of loading garbage."""
        bad = tmp_path / "corrupt.pkl"
        bad.write_bytes(b"this is not a pickle")
        with caplog.at_level(logging.ERROR, logger="app.model"):
            with pytest.raises(pickle.UnpicklingError):
                MovieRatingModel(model_path=str(bad))
        assert "Error loading model" in caplog.text

    def test_model_path_is_stored(self, make_stub_model):
        """The wrapper remembers where it loaded the model from."""
        model = make_stub_model(3.0)
        assert model.model_path.endswith("stub_3.0.pkl")


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
