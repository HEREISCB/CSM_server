# CSM TTS Server - Multi-stage Docker Build
# Production-ready container with CUDA support

# =============================================================================
# Stage 1: Base with CUDA and Python
# =============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS deps

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url=https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Production
# =============================================================================
FROM deps AS production

WORKDIR /app

# Copy application code
COPY . .

# Create directories for runtime
RUN mkdir -p /app/audio /app/config /app/logs

# Set environment variables
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8080
ENV TTS_DEVICE=cuda
ENV TTS_MODEL_PATH=""
ENV TTS_REQUIRE_AUTH=false
ENV TTS_MAX_CONCURRENT=10
ENV TTS_MAX_TEXT_LENGTH=4096
ENV TTS_MAX_AUDIO_MS=120000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run the server
CMD ["python", "server.py"]

# =============================================================================
# Stage 4: Development (optional)
# =============================================================================
FROM production AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    websockets

# Enable reload for development
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
