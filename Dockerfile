# Use a slim Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, FFmpeg, and Sports2D
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requirements: user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    TORCH_HOME=$HOME/app/models \
    MPLCONFIGDIR=$HOME/app/.matplotlib

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO models to speed up container startup on HF
# This ensures the models are baked into the image layers.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"

# Copy the rest of the application
COPY --chown=user . .

# Expose the API port (HF expects 7860 by default)
EXPOSE 7860

# Use uvicorn to serve the FastAPI app
# We use port 7860 to match HF Spaces requirements
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
