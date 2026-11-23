FROM python:3.11-slim

ARG INSTALL_ML=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install system dependencies needed for scikit-learn/numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: heavy ML deps for sentence-transformers (debunk matching) and
# torch/transformers (IndoBERT). Build with:  docker build --build-arg INSTALL_ML=true .
COPY requirements-ml.txt .
RUN if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY . .

# Re-train the model at build time so the container ships with artifacts
RUN python train_model.py

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Jalankan sebagai non-root user (P02). Buat user & grup, beri kepemilikan
# pada direktori aplikasi serta direktori yang perlu ditulis (feedback,
# cache, model) oleh user tersebut.
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser

CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-2} --timeout ${WEB_TIMEOUT:-120}
