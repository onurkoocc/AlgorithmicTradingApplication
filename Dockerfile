# Use TensorFlow GPU image as base with CUDA 12.3 compatibility
FROM tensorflow/tensorflow:2.16.1-gpu

# Set environment variables for performance
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV PYTHONPATH=/app

# Performance optimization flags for TensorFlow
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_THREAD_COUNT=2
ENV TF_USE_CUDNN=1
ENV TF_CUDNN_USE_AUTOTUNE=1
ENV TF_ENABLE_MKL=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    gcc \
    g++ \
    git \
    curl \
    htop \
    bc \
    lsb-release \
    nvidia-cuda-toolkit \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Create directories for application outputs
RUN mkdir -p /app/EnhancedTrainingResults/MonitorLog \
    /app/EnhancedTrainingResults/BacktestLog \
    /app/EnhancedTrainingResults/TradeLog \
    /app/EnhancedTrainingResults/Trades \
    /app/EnhancedTrainingResults/TemperatureLog \
    /app/EnhancedTrainingResults/OptimizationResults \
    /app/data

# Copy application files
COPY *.py /app/
COPY startup.sh /app/

# Create symbolic link for monitor script
RUN cp /app/monitor.py /app/unified_monitor.py && \
    chmod +x /app/startup.sh

# Set entrypoint and default command
ENTRYPOINT ["/bin/bash"]
CMD ["/app/startup.sh"]