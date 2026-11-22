# Aegis Alpha Production Container Image (Python 3.12 on Debian Slim)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal system dependencies required by scientific Python packages and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to maximize Docker build layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy project source code
COPY . /app

# Create runtime directories and run as a non-root user
RUN mkdir -p /app/data /app/models /app/notebooks \
    && useradd -m -u 1000 aegis \
    && chown -R aegis:aegis /app

USER aegis

# Default command runs deterministic offline reproduction; override with --mode online in production
CMD ["python", "main.py", "--mode", "offline"]
