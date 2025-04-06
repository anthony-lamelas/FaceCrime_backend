# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the application code
COPY . .

# Expose port 
EXPOSE 8443

# Start the FastAPI app with Gunicorn and Uvicorn workers
CMD ["uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8443", "--ws", "auto"]
