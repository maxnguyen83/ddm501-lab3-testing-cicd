"""
FastAPI application for Movie Rating Prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, MODEL_VERSION
from app.model import MovieRatingModel
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: MovieRatingModel = None


@app.on_event("startup")
async def startup_event():
    """Load model when application starts."""
    global model
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict movie rating for a user.
    
    Args:
        request: PredictionRequest with user_id and movie_id
        
    Returns:
        PredictionResponse with predicted rating
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        rating = model.predict(request.user_id, request.movie_id)
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=rating,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict movie ratings for multiple user-movie pairs.
    
    Args:
        request: BatchPredictionRequest with list of predictions
        
    Returns:
        BatchPredictionResponse with all predicted ratings
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for item in request.predictions:
            rating = model.predict(item.user_id, item.movie_id)
            results.append(PredictionResponse(
                user_id=item.user_id,
                movie_id=item.movie_id,
                predicted_rating=rating,
                model_version=MODEL_VERSION
            ))
        return BatchPredictionResponse(
            predictions=results,
            total_count=len(results)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": "SVD (Collaborative Filtering)",
        "is_loaded": model is not None and model.is_loaded(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
