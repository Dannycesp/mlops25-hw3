FROM python:3.10-slim

# Set working directory
WORKDIR /home/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /home/src/mage_data \
    && mkdir -p /home/src/models \
    && mkdir -p /home/src/data

# Copy project files
COPY . .

# Set proper permissions for the working directory
RUN chmod -R 755 /home/src

# Expose Mage port
EXPOSE 6789

# Health check command (matches docker-compose.yml)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:6789 || exit 1
