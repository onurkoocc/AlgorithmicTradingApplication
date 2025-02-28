#!/bin/bash
echo "Setting up environment and monitoring..."

# Verify NVIDIA drivers are available
echo "NVIDIA Driver Information:"
nvidia-smi

# Check library paths and availability
echo "Checking CUDA and cuDNN libraries:"
ls -l /usr/local/cuda/lib64 | grep -E "libcudnn|libcublas|libcusolver"

# Verify TensorFlow GPU detection with detailed logging
echo "TensorFlow GPU Detection:"
python3 -c "import tensorflow as tf; tf.get_logger().setLevel('DEBUG'); print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU')); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPU device name:', tf.test.gpu_device_name())"

# Configure TensorFlow memory optimization
python3 -c "import tensorflow as tf; physical_devices = tf.config.list_physical_devices('GPU'); print(f'Found GPUs: {physical_devices}'); [tf.config.experimental.set_memory_growth(device, True) for device in physical_devices if physical_devices]"

# Start monitors in background
echo "Starting monitoring services..."
python3 memory_monitor.py &
python3 temperature_monitor.py &

# Wait for monitoring services to initialize
sleep 3

# Create necessary directories if they don't exist
mkdir -p EnhancedTrainingResults/MemoryLog
mkdir -p EnhancedTrainingResults/SystemLog
mkdir -p EnhancedTrainingResults/BacktestLog
mkdir -p EnhancedTrainingResults/Trades
mkdir -p EnhancedTrainingResults/TemperatureLog
mkdir -p EnhancedTrainingResults/OptimizationResults

# Run the application
echo "Starting Enhanced Crypto Trading System..."
cd /app
python3 main.py