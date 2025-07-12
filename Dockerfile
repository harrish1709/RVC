FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy all source code
COPY . /app

# Ensure pip can install everything
RUN pip install --upgrade pip==23.3.1

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH for local modules like rvc_python
ENV PYTHONPATH=/app

# Expose port
EXPOSE 10000

# Run with Gunicorn in production
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
