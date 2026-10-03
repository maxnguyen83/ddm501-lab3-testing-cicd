"""
Integration tests for API endpoints.

TODO: Complete the test implementations below.

Run tests:
    pytest tests/integration/test_api.py -v
"""

import pytest


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
    # TODO 1: Implement Additional Health Tests
    # =========================================================================
    
    def test_health_response_has_model_loaded_field(self, test_client):
        """
        Test that health response has model_loaded field.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.get("/health")
        # data = response.json()
        # assert "model_loaded" in data
        pass
    
    def test_health_model_loaded_is_boolean(self, test_client):
        """
        Test that model_loaded is a boolean value.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass


class TestRootEndpoint:
    """Integration tests for / endpoint."""
    
    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200 status code."""
        response = test_client.get("/")
        assert response.status_code == 200
    
    # =========================================================================
    # TODO 2: Implement Root Endpoint Tests
    # =========================================================================
    
    def test_root_contains_api_info(self, test_client):
        """
        Test that root response contains API information.
        
        TODO: Implement this test
        - Check for 'name', 'version', 'docs' fields
        """
        # TODO: Implement
        pass


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
    # TODO 3: Implement Response Structure Tests
    # =========================================================================
    
    def test_predict_response_has_predicted_rating(self, test_client, sample_prediction_request):
        """
        Test that response contains predicted_rating field.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.post("/predict", json=sample_prediction_request)
        # data = response.json()
        # assert "predicted_rating" in data
        pass
    
    def test_predict_response_has_user_id(self, test_client, sample_prediction_request):
        """
        Test that response contains user_id field.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    def test_predict_response_has_movie_id(self, test_client, sample_prediction_request):
        """
        Test that response contains movie_id field.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    def test_predict_response_rating_in_valid_range(self, test_client, sample_prediction_request):
        """
        Test that predicted_rating is between 1.0 and 5.0.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    # =========================================================================
    # TODO 4: Implement Validation Error Tests
    # =========================================================================
    
    def test_predict_missing_user_id_returns_422(self, test_client):
        """
        Test that missing user_id returns 422 Unprocessable Entity.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.post("/predict", json={"movie_id": "242"})
        # assert response.status_code == 422
        pass
    
    def test_predict_missing_movie_id_returns_422(self, test_client):
        """
        Test that missing movie_id returns 422.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    def test_predict_empty_body_returns_422(self, test_client):
        """
        Test that empty request body returns 422.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.post("/predict", json={})
        # assert response.status_code == 422
        pass
    
    def test_predict_invalid_json_returns_422(self, test_client):
        """
        Test that invalid JSON returns 422.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    # =========================================================================
    # TODO 5: Implement Multiple Request Tests
    # =========================================================================
    
    def test_predict_multiple_valid_requests(self, test_client, known_user_movie_pairs):
        """
        Test multiple prediction requests all succeed.
        
        TODO: Implement this test
        - Loop through known_user_movie_pairs
        - Make prediction request for each
        - Assert all return 200
        """
        # TODO: Implement
        pass


class TestBatchPredictEndpoint:
    """Integration tests for /predict/batch endpoint."""
    
    # =========================================================================
    # TODO 6: Implement Batch Prediction Tests
    # =========================================================================
    
    def test_batch_predict_returns_200(self, test_client, sample_batch_request):
        """
        Test that batch prediction returns 200.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.post("/predict/batch", json=sample_batch_request)
        # assert response.status_code == 200
        pass
    
    def test_batch_predict_returns_correct_count(self, test_client, sample_batch_request):
        """
        Test that batch prediction returns correct number of results.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    def test_batch_predict_all_ratings_in_range(self, test_client, sample_batch_request):
        """
        Test that all batch predictions are in valid range.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass


class TestErrorHandling:
    """Tests for API error handling."""
    
    # =========================================================================
    # TODO 7: Implement Error Handling Tests
    # =========================================================================
    
    def test_404_for_unknown_endpoint(self, test_client):
        """
        Test that unknown endpoint returns 404.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.get("/unknown")
        # assert response.status_code == 404
        pass
    
    def test_method_not_allowed_get_predict(self, test_client):
        """
        Test that GET /predict returns 405 Method Not Allowed.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # response = test_client.get("/predict")
        # assert response.status_code == 405
        pass
    
    def test_method_not_allowed_post_health(self, test_client):
        """
        Test that POST /health returns 405.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""
    
    def test_model_info_returns_200(self, test_client):
        """Test that model info endpoint returns 200."""
        response = test_client.get("/model/info")
        assert response.status_code == 200
    
    # =========================================================================
    # TODO 8: Implement Model Info Tests
    # =========================================================================
    
    def test_model_info_has_version(self, test_client):
        """
        Test that model info has version field.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    def test_model_info_has_is_loaded(self, test_client):
        """
        Test that model info has is_loaded field.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
