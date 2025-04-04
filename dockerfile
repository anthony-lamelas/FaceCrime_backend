# Use the official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose port 
EXPOSE 8443

# Start the FastAPI app with Gunicorn and Uvicorn workers
CMD ["uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8443", "--ws", "auto"]
