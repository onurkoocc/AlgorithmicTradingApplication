# Use TensorFlow GPU image as base
FROM tensorflow/tensorflow:2.16.1-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV PYTHONPATH=/app

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies from requirements file
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Create directories for application outputs
RUN mkdir -p /app/EnhancedTrainingResults/MemoryLog \
    /app/EnhancedTrainingResults/SystemLog \
    /app/EnhancedTrainingResults/BacktestLog \
    /app/EnhancedTrainingResults/TradeLog \
    /app/EnhancedTrainingResults/Trades \
    /app/EnhancedTrainingResults/TemperatureLog \
    /app/EnhancedTrainingResults/OptimizationResults

# Increase lookback values in Config for better data collection
RUN echo "# Increase lookback values" > /app/config_override.py && \
    echo "from config import Config" >> /app/config_override.py && \
    echo "Config.LOOKBACK_30M_CANDLES = 5000" >> /app/config_override.py && \
    echo "Config.LOOKBACK_4H_CANDLES = 625" >> /app/config_override.py && \
    echo "Config.LOOKBACK_DAILY_CANDLES = 150" >> /app/config_override.py

# Copy application files
COPY startup.sh /app/
COPY config.py data_manager.py feature_engineering.py /app/
COPY data_preparation.py model_management.py trading_logic.py /app/
COPY main.py memory_monitor.py temperature_monitor.py /app/
COPY config_override.py /app/

# Make startup script executable
RUN chmod +x /app/startup.sh

# Create a modified startup script that applies our fixes
RUN echo '#!/bin/bash' > /app/enhanced_startup.sh && \
    echo 'echo "Setting up environment and monitoring..."' >> /app/enhanced_startup.sh && \
    echo 'echo "Applying configuration overrides for improved performance..."' >> /app/enhanced_startup.sh && \
    echo 'python3 -c "import config_override"' >> /app/enhanced_startup.sh && \
    echo 'echo "Starting the Enhanced Crypto Trading System..."' >> /app/enhanced_startup.sh && \
    echo 'cd /app' >> /app/enhanced_startup.sh && \
    echo 'python3 main.py' >> /app/enhanced_startup.sh && \
    chmod +x /app/enhanced_startup.sh

# Set entrypoint and default command
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/app/enhanced_startup.sh"]