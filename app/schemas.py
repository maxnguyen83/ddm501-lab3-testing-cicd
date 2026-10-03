"""
Pydantic schemas for request/response validation.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_RATING, MIN_RATING

# Input bounds. IDs in MovieLens are short numeric strings; 50 characters is
# generous and stops oversized payloads before they reach the model.
MAX_ID_LENGTH = 50
MAX_BATCH_SIZE = 100


class _UserMovieIds(BaseModel):
    """Shared user/movie ID fields and validation for single and batch requests."""

    user_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH, examples=["196"])
    movie_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH, examples=["242"])

    @field_validator("user_id", "movie_id")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that IDs are not empty or whitespace only."""
        if not v.strip():
            raise ValueError("ID cannot be empty or whitespace only")
        return v.strip()


class PredictionRequest(_UserMovieIds):
    """Request schema for prediction endpoint."""


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    # Field names starting with "model_" clash with pydantic's reserved prefix
    model_config = ConfigDict(protected_namespaces=())

    user_id: str
    movie_id: str
    predicted_rating: float = Field(..., ge=MIN_RATING, le=MAX_RATING)
    model_version: str


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class PredictionItem(_UserMovieIds):
    """Single prediction item for batch requests."""


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""

    predictions: List[PredictionItem] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""

    predictions: List[PredictionResponse]
    total_count: int


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    detail: str
    error_code: str = "UNKNOWN_ERROR"
