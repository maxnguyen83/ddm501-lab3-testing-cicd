"""
Unit tests for Pydantic schemas.

Run tests:
    pytest tests/unit/test_schemas.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    MAX_BATCH_SIZE,
    MAX_ID_LENGTH,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


def _response(rating: float) -> PredictionResponse:
    """Build a PredictionResponse with fixed IDs and the given rating."""
    return PredictionResponse(
        user_id="196", movie_id="242", predicted_rating=rating, model_version="1.0.0"
    )


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    # =========================================================================
    # Valid Input Tests (PROVIDED)
    # =========================================================================

    def test_valid_request(self):
        """Test that valid request passes validation."""
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self):
        """Test numeric string IDs are valid."""
        request = PredictionRequest(user_id="123", movie_id="456")
        assert request.user_id == "123"
        assert request.movie_id == "456"

    # =========================================================================
    # Missing Field Tests
    # =========================================================================

    def test_missing_user_id_raises_error(self):
        """Test that missing user_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(movie_id="242")
        assert exc_info.value.errors()[0]["loc"] == ("user_id",)

    def test_missing_movie_id_raises_error(self):
        """Test that missing movie_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(user_id="196")
        assert exc_info.value.errors()[0]["loc"] == ("movie_id",)

    def test_missing_both_fields_raises_error(self):
        """Test that missing both fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest()
        assert exc_info.value.error_count() == 2

    # =========================================================================
    # Empty/Invalid Input Tests
    # =========================================================================

    def test_empty_user_id_raises_error(self):
        """Test that empty user_id raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    def test_whitespace_only_user_id_raises_error(self):
        """Test that whitespace-only user_id raises ValidationError."""
        with pytest.raises(ValidationError, match="empty or whitespace"):
            PredictionRequest(user_id="   ", movie_id="242")

    def test_whitespace_only_movie_id_raises_error(self):
        """Whitespace-only movie_id is rejected the same way."""
        with pytest.raises(ValidationError, match="empty or whitespace"):
            PredictionRequest(user_id="196", movie_id="\t\n")

    def test_none_values_raise_error(self):
        """Test that None values raise ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=None, movie_id="242")
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196", movie_id=None)

    def test_surrounding_whitespace_is_stripped(self):
        """IDs are normalised so ' 196 ' and '196' hit the same user."""
        request = PredictionRequest(user_id=" 196 ", movie_id="242\n")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    # =========================================================================
    # Length Bound Tests (oversized payload protection)
    # =========================================================================

    def test_id_at_max_length_is_accepted(self):
        """An ID of exactly MAX_ID_LENGTH characters is valid."""
        request = PredictionRequest(user_id="1" * MAX_ID_LENGTH, movie_id="242")
        assert len(request.user_id) == MAX_ID_LENGTH

    def test_id_over_max_length_raises_error(self):
        """One character over the bound is rejected."""
        with pytest.raises(ValidationError, match="at most 50 characters"):
            PredictionRequest(user_id="1" * (MAX_ID_LENGTH + 1), movie_id="242")

    def test_very_large_id_raises_error(self):
        """A 10,000-character ID is rejected by validation."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="1" * 10_000, movie_id="242")

    def test_length_bound_applies_before_stripping(self):
        """Padding with spaces cannot be used to sneak a huge payload through."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=" " * 10_000 + "196", movie_id="242")

    # =========================================================================
    # Type Validation Tests
    # =========================================================================

    def test_integer_user_id_converted_to_string(self):
        """
        Pydantic v2 does not coerce int to str by default, so an integer
        user_id is rejected rather than converted.
        """
        with pytest.raises(ValidationError, match="string_type|valid string"):
            PredictionRequest(user_id=196, movie_id="242")

    def test_integer_ids_both_rejected(self):
        """Both integer IDs produce one error each."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(user_id=196, movie_id=242)
        assert exc_info.value.error_count() == 2

    def test_list_id_rejected(self):
        """Structured values are not valid IDs."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=["196"], movie_id="242")


class TestPredictionItem:
    """Tests for PredictionItem (one entry of a batch request)."""

    def test_valid_item(self):
        """A normal item passes validation."""
        item = PredictionItem(user_id="196", movie_id="242")
        assert (item.user_id, item.movie_id) == ("196", "242")

    def test_whitespace_only_item_rejected(self):
        """
        Regression: PredictionItem used to skip the whitespace check that
        PredictionRequest has, so the batch endpoint accepted '   ' as an ID.
        """
        with pytest.raises(ValidationError, match="empty or whitespace"):
            PredictionItem(user_id="   ", movie_id="242")

    def test_item_ids_are_stripped(self):
        """Regression: batch IDs are stripped like single-request IDs."""
        item = PredictionItem(user_id=" 196 ", movie_id=" 242")
        assert (item.user_id, item.movie_id) == ("196", "242")

    def test_item_and_request_validate_identically(self):
        """Single and batch inputs follow the same rules."""
        for raw in ["196", " 196 ", "x" * MAX_ID_LENGTH]:
            assert PredictionItem(user_id=raw, movie_id="1").user_id == (
                PredictionRequest(user_id=raw, movie_id="1").user_id
            )
        for bad in ["", "   ", "x" * (MAX_ID_LENGTH + 1)]:
            with pytest.raises(ValidationError):
                PredictionItem(user_id=bad, movie_id="1")
            with pytest.raises(ValidationError):
                PredictionRequest(user_id=bad, movie_id="1")


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """Test that valid response passes validation."""
        response = _response(3.5)
        assert response.predicted_rating == 3.5
        assert response.model_version == "1.0.0"

    def test_rating_below_minimum_raises_error(self):
        """Test that rating below 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _response(0.5)

    def test_rating_above_maximum_raises_error(self):
        """Test that rating above 5.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _response(5.01)

    def test_rating_at_boundaries(self):
        """Test ratings at exact boundaries (1.0 and 5.0)."""
        assert _response(1.0).predicted_rating == 1.0
        assert _response(5.0).predicted_rating == 5.0

    def test_missing_model_version_raises_error(self):
        """model_version is required so clients can trace which model answered."""
        with pytest.raises(ValidationError):
            PredictionResponse(user_id="196", movie_id="242", predicted_rating=3.0)

    def test_response_serialises_to_expected_keys(self):
        """The JSON contract has exactly these four fields."""
        assert set(_response(3.0).model_dump()) == {
            "user_id",
            "movie_id",
            "predicted_rating",
            "model_version",
        }


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health_response(self):
        """Test that valid health response passes validation."""
        health = HealthResponse(status="healthy", model_loaded=True)
        assert health.status == "healthy"
        assert health.model_loaded is True

    def test_health_response_status_types(self):
        """Test various status values."""
        for status, loaded in [("healthy", True), ("unhealthy", False), ("degraded", True)]:
            health = HealthResponse(status=status, model_loaded=loaded)
            assert health.status == status
            assert health.model_loaded is loaded

    def test_health_response_requires_model_loaded(self):
        """model_loaded is mandatory."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy")

    def test_health_response_rejects_non_boolean_model_loaded(self):
        """A value that is not bool-like is rejected."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", model_loaded="maybe")


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    def test_valid_batch_request(self):
        """Test that valid batch request passes validation."""
        batch = BatchPredictionRequest(
            predictions=[
                {"user_id": "196", "movie_id": "242"},
                {"user_id": "186", "movie_id": "302"},
            ]
        )
        assert len(batch.predictions) == 2
        assert all(isinstance(p, PredictionItem) for p in batch.predictions)

    def test_empty_predictions_list_raises_error(self):
        """Test that empty predictions list raises ValidationError."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_max_batch_size_is_accepted(self):
        """Exactly MAX_BATCH_SIZE items is allowed."""
        items = [{"user_id": "196", "movie_id": "242"}] * MAX_BATCH_SIZE
        assert len(BatchPredictionRequest(predictions=items).predictions) == MAX_BATCH_SIZE

    def test_too_many_predictions_raises_error(self):
        """Test that too many predictions raises ValidationError (max_length=100)."""
        items = [{"user_id": "196", "movie_id": "242"}] * (MAX_BATCH_SIZE + 1)
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=items)

    def test_invalid_item_inside_batch_raises_error(self):
        """One bad item fails the whole batch and the error points to it."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(
                predictions=[
                    {"user_id": "196", "movie_id": "242"},
                    {"user_id": "   ", "movie_id": "242"},
                ]
            )
        assert exc_info.value.errors()[0]["loc"][:2] == ("predictions", 1)


class TestOtherSchemas:
    """Tests for the remaining response schemas."""

    def test_schemas_import_without_warnings(self):
        """
        Regression: 'model_version' and 'model_loaded' triggered pydantic's
        protected-namespace UserWarning on every import. A fresh interpreter
        with warnings turned into errors must import the module cleanly.
        """
        result = subprocess.run(
            [sys.executable, "-W", "error", "-c", "import app.schemas"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 0, result.stderr

    def test_batch_response(self):
        """BatchPredictionResponse holds the predictions and their count."""
        batch = BatchPredictionResponse(predictions=[_response(3.0)], total_count=1)
        assert batch.total_count == len(batch.predictions)

    def test_error_response_default_code(self):
        """ErrorResponse falls back to a generic error code."""
        error = ErrorResponse(detail="boom")
        assert error.error_code == "UNKNOWN_ERROR"


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
