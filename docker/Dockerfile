# ============================================================
# RadiologyAI – Backend Dockerfile
# ============================================================
# Multi-stage build: builder installs deps, runtime is lean

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into isolated location
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# System runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project source
COPY . .

# Create necessary directories
RUN mkdir -p \
    backend/uploads \
    backend/uploads/gradcam \
    ai_model/checkpoints \
    datasets/processed \
    datasets/raw \
    logs/tensorboard

# Non-root user for security
RUN useradd -m -u 1000 radai && chown -R radai /app
USER radai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
