"""
ML Model wrapper for movie rating prediction.
"""

import logging
import pickle
from typing import Any, List, Optional, Tuple

from app.config import MAX_RATING, MIN_RATING, MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clean_id(name: str, value: Any) -> str:
    """
    Validate and normalise a raw user/movie ID.

    Surprise looks IDs up as raw strings, so an int (196) or None would be
    treated as an unknown user and silently get the cold-start fallback.
    Rejecting them here turns that silent wrong answer into a clear error.

    Args:
        name: Field name used in the error message
        value: Raw ID passed by the caller

    Returns:
        The ID with surrounding whitespace removed
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} cannot be empty or whitespace only")
    return cleaned


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path = model_path
        self.model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)

        Returns:
            Predicted rating (float between 1.0 and 5.0)

        Raises:
            RuntimeError: If the model is not loaded
            TypeError: If an ID is not a string
            ValueError: If an ID is empty or whitespace only
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        uid = _clean_id("user_id", user_id)
        iid = _clean_id("movie_id", movie_id)

        prediction = self.model.predict(uid, iid)
        rating = round(float(prediction.est), 2)

        # Clip to valid range
        rating = max(MIN_RATING, min(MAX_RATING, rating))

        return rating

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Singleton instance
_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance


def reset_model() -> None:
    """Reset the model instance (useful for testing)."""
    global _model_instance
    _model_instance = None
