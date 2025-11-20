FROM python:3.8-slim

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage cache
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Set PYTHONPATH to include /app so imports like 'from src.detector' work
ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python", "src/run_detection.py"]
