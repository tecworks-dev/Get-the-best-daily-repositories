# Multi-stage build for authenticated model downloads
FROM python:3.10-slim AS model-downloader

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface_hub

# Set working directory
WORKDIR /model-downloader

# Create directory for downloaded models
RUN mkdir -p /model-downloader/models

# This will run when building the image
# You'll need to pass your Hugging Face token at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Login and download model
RUN if [ -n "$HF_TOKEN" ]; then \
    huggingface-cli login --token ${HF_TOKEN}; \
    huggingface-cli download sesame/csm-1b ckpt.pt --local-dir /model-downloader/models; \
    else echo "No HF_TOKEN provided, model download will be skipped"; fi

# Now for the main application stage
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchaudio numpy

# Install torchao from source
RUN pip3 install --no-cache-dir git+https://github.com/pytorch/ao.git

# Install torchtune from source with specific branch for latest features
RUN git clone https://github.com/pytorch/torchtune.git /tmp/torchtune && \
    cd /tmp/torchtune && \
    # Try to use the main branch, which should have llama3_2
    git checkout main && \
    pip install -e .

# Install remaining dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app /app/app

# Create and set up models directory
RUN mkdir -p /app/models

# Copy downloaded model from the model-downloader stage
COPY --from=model-downloader /model-downloader/models /app/models

# Show available models in torchtune
RUN python3 -c "import torchtune.models; print('Available models in torchtune:', dir(torchtune.models))"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3", "-m", "app.main"]