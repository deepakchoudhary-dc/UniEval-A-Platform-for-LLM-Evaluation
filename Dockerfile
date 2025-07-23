# Use Python 3.11 slim image for better compatibility
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout and no cache
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p src/data/memory src/data/search logs data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
