# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps (OpenCV headless needs libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/  ./src/
COPY api/  ./api/
COPY ui/   ./ui/

# Create runtime directories (volumes will overlay these)
RUN mkdir -p data/train data/test models

# Default: run FastAPI
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
