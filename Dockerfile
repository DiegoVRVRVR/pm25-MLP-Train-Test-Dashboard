# ===== Multi-stage Dockerfile for PM2.5 MLP Training Dashboard =====

# ===== Stage 1: Build Stage =====
FROM python:3.12.7-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ===== Stage 2: Runtime Stage =====
FROM python:3.12.7-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set work directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads models static/css static/js templates

# Set permissions
RUN chown -R appuser:appuser /app
RUN chmod -R 755 /app
RUN chmod -R 775 /app/uploads /app/logs /app/models

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Switch to non-root user
USER appuser

# Run the application
CMD ["gunicorn", "app_flask:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120"]