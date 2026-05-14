# ─────────────────────────────────────────────────────────────
# Dockerfile — Loan Propensity FastAPI
# Multi-stage build: keeps final image lean (~400MB)
# ─────────────────────────────────────────────────────────────

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /install
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install/packages --no-cache-dir -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.11-slim
LABEL maintainer="Data Science Team"
LABEL description="Loan Payment Propensity Prediction API"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# Copy application code
COPY src/          ./src/
COPY api/          ./api/
COPY config/       ./config/
COPY artifacts/    ./artifacts/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 5000

# Health check — AWS ELB liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn for production Flask serving (not Flask dev server)
CMD ["gunicorn", "api.flask_app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
