"""
Configuration settings for the Movie Rating API.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "svd_model.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# API settings
API_TITLE = "Movie Rating Prediction API"
API_DESCRIPTION = "API for predicting movie ratings using collaborative filtering"
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Rating constraints
MIN_RATING = 1.0
MAX_RATING = 5.0
