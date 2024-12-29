# Use an official Python runtime as a base image
# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --fix-missing --no-install-recommends \
    libgl1-mesa-glx \
    python3-pip \
    python3 \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.nvidia.com

# Copy the entire source code and configs
COPY ./src /app/src
COPY ./configs /app/configs
COPY ./weights /app/weights

# Set the default command for running FastAPI or inference
#CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
