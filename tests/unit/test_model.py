"""
Unit tests for MovieRatingModel class.

TODO: Complete the test implementations below.

Run tests:
    pytest tests/unit/test_model.py -v
"""

import pytest
from app.model import MovieRatingModel


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
    # TODO 1: Implement Prediction Return Type Tests
    # =========================================================================
    
    def test_predict_returns_float(self, trained_model):
        """
        Test that predict() returns a float value.
        
        TODO: Implement this test
        - Call trained_model.predict() with valid user_id and movie_id
        - Assert that the result is an instance of float
        """
        # TODO: Implement
        # result = trained_model.predict("196", "242")
        # assert isinstance(result, float)
        pass
    
    # =========================================================================
    # TODO 2: Implement Rating Range Tests
    # =========================================================================
    
    def test_predict_returns_value_in_valid_range(self, trained_model):
        """
        Test that predictions are within 1-5 range.
        
        TODO: Implement this test
        - Call predict() with a valid user-movie pair
        - Assert that result is >= 1.0 and <= 5.0
        """
        # TODO: Implement
        # result = trained_model.predict("196", "242")
        # assert 1.0 <= result <= 5.0
        pass
    
    def test_predict_multiple_pairs_all_in_range(self, trained_model, known_user_movie_pairs):
        """
        Test that all predictions are in valid range.
        
        TODO: Implement this test
        - Loop through known_user_movie_pairs
        - For each pair, call predict() and verify range
        """
        # TODO: Implement
        # for pair in known_user_movie_pairs:
        #     result = trained_model.predict(pair["user_id"], pair["movie_id"])
        #     assert 1.0 <= result <= 5.0
        pass
    
    # =========================================================================
    # TODO 3: Implement Batch Prediction Tests
    # =========================================================================
    
    def test_predict_batch_returns_list(self, trained_model):
        """
        Test that predict_batch() returns a list.
        
        TODO: Implement this test
        - Create list of (user_id, movie_id) tuples
        - Call predict_batch()
        - Assert result is a list
        """
        # TODO: Implement
        # pairs = [("196", "242"), ("186", "302")]
        # results = trained_model.predict_batch(pairs)
        # assert isinstance(results, list)
        pass
    
    def test_predict_batch_returns_correct_length(self, trained_model):
        """
        Test that predict_batch() returns correct number of results.
        
        TODO: Implement this test
        - Create list of pairs
        - Assert len(results) == len(pairs)
        """
        # TODO: Implement
        # pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        # results = trained_model.predict_batch(pairs)
        # assert len(results) == len(pairs)
        pass
    
    def test_predict_batch_all_values_in_range(self, trained_model):
        """
        Test that all batch predictions are in valid range.
        
        TODO: Implement this test
        """
        # TODO: Implement
        pass
    
    # =========================================================================
    # TODO 4: Implement is_loaded() Tests
    # =========================================================================
    
    def test_is_loaded_returns_bool(self, trained_model):
        """
        Test that is_loaded() returns a boolean.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # result = trained_model.is_loaded()
        # assert isinstance(result, bool)
        pass
    
    def test_is_loaded_returns_true_for_loaded_model(self, trained_model):
        """
        Test that is_loaded() returns True for loaded model.
        
        TODO: Implement this test
        """
        # TODO: Implement
        # assert trained_model.is_loaded() == True
        pass
    
    # =========================================================================
    # TODO 5: Implement Error Handling Tests (BONUS)
    # =========================================================================
    
    def test_predict_with_none_user_id(self, trained_model):
        """
        Test behavior when user_id is None.
        
        TODO: Implement this test (BONUS)
        - This might raise an exception or return a default value
        """
        # TODO: Implement
        pass
    
    def test_predict_with_empty_string(self, trained_model):
        """
        Test behavior when IDs are empty strings.
        
        TODO: Implement this test (BONUS)
        """
        # TODO: Implement
        pass


class TestModelFileHandling:
    """Tests for model file handling."""
    
    def test_model_raises_error_for_missing_file(self):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="/nonexistent/path/model.pkl")


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
