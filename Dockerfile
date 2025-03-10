# Multi-stage build for optimized caching
# Stage 1: Build the base image with all dependencies
FROM tensorflow/tensorflow:2.16.1-gpu AS base

# Set environment variables for stability and performance
ENV TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_GPU_ALLOCATOR=cuda_malloc_async \
    TF_GPU_THREAD_MODE=gpu_private \
    TF_GPU_THREAD_COUNT=2 \
    TF_ENABLE_ONEDNN_OPTS=1 \
    TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=1 \
    TF_ENABLE_CUDA_MALLOC_ASYNC=1 \
    TF_XLA_FLAGS="--tf_xla_auto_jit=2" \
    CRYPTO_SYSTEM_GPU_MEMORY_LIMIT_PCT=65

# Set working directory
WORKDIR /app

# Install system dependencies (this layer changes rarely)
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
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies in a single layer
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir nvidia-tensorrt pynvml

# Create directories for application outputs
RUN mkdir -p /app/EnhancedTrainingResults/MonitorLog \
    /app/EnhancedTrainingResults/BacktestLog \
    /app/EnhancedTrainingResults/TradeLog \
    /app/EnhancedTrainingResults/Trades \
    /app/EnhancedTrainingResults/TemperatureLog \
    /app/EnhancedTrainingResults/OptimizationResults \
    /app/data

# Stage 2: Build the final image with application code
FROM base AS app

# Copy configuration files (these change less frequently)
COPY optimal_params.json /app/
COPY config_unified.py /app/

# Copy source code (these change most frequently)
COPY api_security.py data_manager.py data_preparation.py \
     feature_engineering.py main.py model_management.py \
     memory_utils.py trading_logic.py unified_monitor.py \
     memory_tester.py /app/

# Copy and prepare startup script
COPY startup.sh /app/
RUN chmod +x /app/startup.sh && \
    sed -i 's/python /python3 /g' /app/startup.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
CMD ["/app/startup.sh"]