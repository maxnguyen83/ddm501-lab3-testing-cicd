# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 3: Testing & CI/CD
# =============================================================================

FROM python:3.10-slim

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

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
