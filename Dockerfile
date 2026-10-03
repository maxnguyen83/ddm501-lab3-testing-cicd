# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 3: Testing & CI/CD
#
# The image bakes in models/svd_model.pkl. In CI/CD the model is trained and
# tested first, then copied into the build context, so the image ships the
# exact artefact that passed the tests.
# =============================================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (for cache optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check: python:3.10-slim has no curl, and /health answers 200 even
# without a model, so check the model_loaded flag in the JSON body instead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import json,sys,urllib.request; r=urllib.request.urlopen('http://localhost:8000/health', timeout=5); sys.exit(0 if json.load(r).get('model_loaded') else 1)"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
