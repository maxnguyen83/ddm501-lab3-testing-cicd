"""
Integration tests for API endpoints.

These tests drive the real FastAPI app (routing, validation, lifespan model
loading, serialisation) through the TestClient with the trained model.

Run tests:
    pytest tests/integration/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import MODEL_VERSION
from app.schemas import MAX_BATCH_SIZE, MAX_ID_LENGTH


@pytest.fixture
def model_unloaded(monkeypatch):
    """Simulate a service whose model failed to load (restored after the test)."""
    monkeypatch.setattr(main, "model", None)


@pytest.fixture
def failing_model(monkeypatch, test_client):
    """Make every prediction raise an internal error containing a 'secret'."""

    def boom(user_id, movie_id):
        raise RuntimeError("internal detail: /srv/models/svd_model.pkl corrupted")

    monkeypatch.setattr(main.model, "predict", boom)


class TestHealthEndpoint:
    """Integration tests for /health endpoint."""

    # =========================================================================
    # Provided Tests
    # =========================================================================

    def test_health_returns_200(self, test_client):
        """Test that health endpoint returns 200 status code."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, test_client):
        """Test that health response has status field."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data

    # =========================================================================
    # Additional Health Tests
    # =========================================================================

    def test_health_response_has_model_loaded_field(self, test_client):
        """Test that health response has model_loaded field."""
        response = test_client.get("/health")
        data = response.json()
        assert "model_loaded" in data

    def test_health_model_loaded_is_boolean(self, test_client):
        """Test that model_loaded is a boolean value."""
        data = test_client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)

    def test_health_reports_healthy_when_model_loaded(self, test_client):
        """With the trained model present the lifespan loads it at startup."""
        data = test_client.get("/health").json()
        assert data == {"status": "healthy", "model_loaded": True}

    def test_health_reports_unhealthy_without_model(self, test_client, model_unloaded):
        """Without a model the endpoint still answers but says unhealthy."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "unhealthy", "model_loaded": False}


class TestRootEndpoint:
    """Integration tests for / endpoint."""

    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200 status code."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self, test_client):
        """Test that root response contains API information."""
        data = test_client.get("/").json()
        for field in ("name", "version", "docs", "health"):
            assert field in data
        assert data["docs"] == "/docs"

    def test_openapi_schema_lists_prediction_routes(self, test_client):
        """The generated OpenAPI document exposes the public routes."""
        paths = test_client.get("/openapi.json").json()["paths"]
        assert {"/predict", "/predict/batch", "/health", "/model/info"} <= set(paths)


class TestPredictEndpoint:
    """Integration tests for /predict endpoint."""

    # =========================================================================
    # Provided Tests
    # =========================================================================

    def test_predict_valid_request_returns_200(self, test_client, sample_prediction_request):
        """Test that valid prediction request returns 200."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

    # =========================================================================
    # Response Structure Tests
    # =========================================================================

    def test_predict_response_has_predicted_rating(self, test_client, sample_prediction_request):
        """Test that response contains predicted_rating field."""
        response = test_client.post("/predict", json=sample_prediction_request)
        data = response.json()
        assert "predicted_rating" in data
        assert isinstance(data["predicted_rating"], float)

    def test_predict_response_has_user_id(self, test_client, sample_prediction_request):
        """Test that response contains user_id field."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(self, test_client, sample_prediction_request):
        """Test that response contains movie_id field."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_rating_in_valid_range(self, test_client, sample_prediction_request):
        """Test that predicted_rating is between 1.0 and 5.0."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_response_has_model_version(self, test_client, sample_prediction_request):
        """Responses are traceable to the configured model version."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["model_version"] == MODEL_VERSION

    def test_api_matches_model_wrapper(self, test_client, trained_model):
        """The API returns exactly what the model wrapper predicts."""
        data = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"}).json()
        assert data["predicted_rating"] == trained_model.predict("196", "242")

    def test_predict_strips_whitespace_around_ids(self, test_client):
        """' 196 ' is the same user as '196'."""
        padded = test_client.post("/predict", json={"user_id": " 196 ", "movie_id": "242 "})
        clean = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        assert padded.status_code == 200
        assert padded.json() == clean.json()

    def test_predict_unknown_user_uses_fallback(self, test_client):
        """A cold-start user still gets a valid rating (no 4xx/5xx)."""
        response = test_client.post("/predict", json={"user_id": "99999", "movie_id": "242"})
        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    # =========================================================================
    # Validation Error Tests
    # =========================================================================

    def test_predict_missing_user_id_returns_422(self, test_client):
        """Test that missing user_id returns 422 Unprocessable Entity."""
        response = test_client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422

    def test_predict_missing_movie_id_returns_422(self, test_client):
        """Test that missing movie_id returns 422."""
        response = test_client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "movie_id"]

    def test_predict_empty_body_returns_422(self, test_client):
        """Test that empty request body returns 422."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client):
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_integer_ids_return_422(self, test_client):
        """JSON numbers are not accepted as IDs (they must be strings)."""
        response = test_client.post("/predict", json={"user_id": 196, "movie_id": 242})
        assert response.status_code == 422

    def test_all_invalid_requests_rejected(self, test_client, invalid_prediction_requests):
        """Every request in the shared invalid-input fixture is rejected with 422."""
        for payload in invalid_prediction_requests:
            response = test_client.post("/predict", json=payload)
            assert response.status_code == 422, payload

    def test_id_at_length_bound_accepted(self, test_client):
        """Boundary: an ID of exactly MAX_ID_LENGTH characters is processed."""
        response = test_client.post(
            "/predict", json={"user_id": "1" * MAX_ID_LENGTH, "movie_id": "242"}
        )
        assert response.status_code == 200

    # =========================================================================
    # Multiple Request Tests
    # =========================================================================

    def test_predict_multiple_valid_requests(self, test_client, known_user_movie_pairs):
        """Test multiple prediction requests all succeed."""
        for pair in known_user_movie_pairs:
            payload = {"user_id": pair["user_id"], "movie_id": pair["movie_id"]}
            response = test_client.post("/predict", json=payload)
            assert response.status_code == 200, payload
            assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    # =========================================================================
    # Service Error Tests
    # =========================================================================

    def test_predict_returns_503_when_model_not_loaded(self, test_client, model_unloaded):
        """No model means 503 Service Unavailable, not a crash."""
        response = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_predict_internal_error_returns_generic_500(self, test_client, failing_model):
        """Internal errors return 500 without leaking the exception text."""
        response = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Prediction failed"
        assert "svd_model.pkl" not in response.text


class TestBatchPredictEndpoint:
    """Integration tests for /predict/batch endpoint."""

    def test_batch_predict_returns_200(self, test_client, sample_batch_request):
        """Test that batch prediction returns 200."""
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200

    def test_batch_predict_returns_correct_count(self, test_client, sample_batch_request):
        """Test that batch prediction returns correct number of results."""
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        expected = len(sample_batch_request["predictions"])
        assert data["total_count"] == expected
        assert len(data["predictions"]) == expected

    def test_batch_predict_all_ratings_in_range(self, test_client, sample_batch_request):
        """Test that all batch predictions are in valid range."""
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert all(1.0 <= p["predicted_rating"] <= 5.0 for p in data["predictions"])

    def test_batch_preserves_request_order(self, test_client, sample_batch_request):
        """Result i belongs to request item i."""
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        sent = [(p["user_id"], p["movie_id"]) for p in sample_batch_request["predictions"]]
        got = [(p["user_id"], p["movie_id"]) for p in data["predictions"]]
        assert got == sent

    def test_batch_matches_single_predictions(self, test_client, sample_batch_request):
        """Batch and single endpoints agree for the same pairs."""
        batch = test_client.post("/predict/batch", json=sample_batch_request).json()
        for item, result in zip(sample_batch_request["predictions"], batch["predictions"]):
            single = test_client.post("/predict", json=item).json()
            assert result == single

    def test_batch_empty_list_returns_422(self, test_client):
        """An empty batch is a client error."""
        response = test_client.post("/predict/batch", json={"predictions": []})
        assert response.status_code == 422

    def test_batch_at_max_size_returns_200(self, test_client):
        """Boundary: exactly MAX_BATCH_SIZE items is accepted."""
        items = [{"user_id": "196", "movie_id": "242"}] * MAX_BATCH_SIZE
        response = test_client.post("/predict/batch", json={"predictions": items})
        assert response.status_code == 200
        assert response.json()["total_count"] == MAX_BATCH_SIZE

    def test_batch_over_max_size_returns_422(self, test_client):
        """One item over the limit is rejected before any prediction runs."""
        items = [{"user_id": "196", "movie_id": "242"}] * (MAX_BATCH_SIZE + 1)
        response = test_client.post("/predict/batch", json={"predictions": items})
        assert response.status_code == 422

    def test_batch_whitespace_id_returns_422(self, test_client):
        """
        Regression: batch items used to accept whitespace-only IDs, which the
        single endpoint rejects.
        """
        payload = {"predictions": [{"user_id": "   ", "movie_id": "242"}]}
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_strips_whitespace_like_single(self, test_client):
        """
        Regression: ' 196' in a batch used to be looked up verbatim, so the
        user was treated as unknown and got a different rating than /predict.
        """
        single = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"}).json()
        payload = {"predictions": [{"user_id": " 196", "movie_id": "242 "}]}
        batch = test_client.post("/predict/batch", json=payload).json()
        assert batch["predictions"][0] == single

    def test_batch_returns_503_when_model_not_loaded(self, test_client, model_unloaded):
        """Batch endpoint reports 503 when the model is missing."""
        response = test_client.post(
            "/predict/batch", json={"predictions": [{"user_id": "196", "movie_id": "242"}]}
        )
        assert response.status_code == 503

    def test_batch_internal_error_returns_generic_500(self, test_client, failing_model):
        """Batch internal errors are not leaked either."""
        response = test_client.post(
            "/predict/batch", json={"predictions": [{"user_id": "196", "movie_id": "242"}]}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Batch prediction failed"
        assert "svd_model.pkl" not in response.text


class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_for_unknown_endpoint(self, test_client):
        """Test that unknown endpoint returns 404."""
        response = test_client.get("/unknown")
        assert response.status_code == 404

    def test_method_not_allowed_get_predict(self, test_client):
        """Test that GET /predict returns 405 Method Not Allowed."""
        response = test_client.get("/predict")
        assert response.status_code == 405

    def test_method_not_allowed_post_health(self, test_client):
        """Test that POST /health returns 405."""
        response = test_client.post("/health")
        assert response.status_code == 405

    def test_method_not_allowed_get_batch(self, test_client):
        """GET /predict/batch is not a valid route either."""
        response = test_client.get("/predict/batch")
        assert response.status_code == 405

    def test_large_payload_rejected(self, test_client):
        """Test that extremely large payloads are rejected."""
        large_payload = {"user_id": "1" * 10000, "movie_id": "242"}
        response = test_client.post("/predict", json=large_payload)
        # Should be rejected by validation
        assert response.status_code in [400, 422]

    def test_large_payload_in_batch_rejected(self, test_client):
        """An oversized ID inside a batch item is rejected too."""
        payload = {"predictions": [{"user_id": "196", "movie_id": "2" * 10_000}]}
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_wrong_content_type_rejected(self, test_client):
        """Form-encoded data is not accepted where JSON is expected."""
        response = test_client.post("/predict", data={"user_id": "196", "movie_id": "242"})
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self, test_client):
        """Test that model info endpoint returns 200."""
        response = test_client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_version(self, test_client):
        """Test that model info has version field."""
        data = test_client.get("/model/info").json()
        assert data["model_version"] == MODEL_VERSION

    def test_model_info_has_is_loaded(self, test_client):
        """Test that model info has is_loaded field."""
        data = test_client.get("/model/info").json()
        assert data["is_loaded"] is True
        assert "SVD" in data["model_type"]

    def test_model_info_reflects_missing_model(self, test_client, model_unloaded):
        """is_loaded flips to False when the model is gone."""
        assert test_client.get("/model/info").json()["is_loaded"] is False


class TestStartup:
    """Tests for the lifespan (startup) behaviour."""

    def test_startup_survives_missing_model(self, monkeypatch):
        """
        If the model file cannot be loaded the app still starts, reports
        unhealthy and answers predictions with 503.
        """

        def missing_model():
            raise FileNotFoundError("models/svd_model.pkl")

        monkeypatch.setattr(main, "model", main.model)  # restore after the test
        monkeypatch.setattr(main, "MovieRatingModel", missing_model)
        with TestClient(main.app) as client:
            assert client.get("/health").json()["model_loaded"] is False
            response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
            assert response.status_code == 503

    def test_startup_loads_model(self, monkeypatch):
        """Entering the client context runs the lifespan and loads the model."""
        monkeypatch.setattr(main, "model", None)
        with TestClient(main.app) as client:
            assert client.get("/health").json()["model_loaded"] is True
            assert main.model is not None and main.model.is_loaded()


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
