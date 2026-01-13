# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . /app

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app && chown -R app:app /app
USER app

# Command to run the application
CMD ["python", "run_hf_spaces.py"]