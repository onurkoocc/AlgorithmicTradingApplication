#!/bin/bash
echo "Setting up optimized environment for RTX 4070..."

# Verify Python installation and path
echo "Python path verification:"
which python3
python3 --version

# Verify NVIDIA drivers are available
echo "NVIDIA Driver Information:"
nvidia-smi || echo "Warning: NVIDIA driver not available"

GPU_MEMORY_LIMIT=${CRYPTO_SYSTEM_GPU_MEMORY_LIMIT_PCT:-65}

# Configure GPU with consistent approach
python3 -c "
import tensorflow as tf
import os

# Use environment variable for memory limit
gpu_memory_limit = int(os.environ.get('CRYPTO_SYSTEM_GPU_MEMORY_LIMIT_PCT', 70))
print(f'Using GPU memory limit: {gpu_memory_limit}%')

# Configure GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f'Found GPUs: {physical_devices}')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f'Enabled memory growth for {device}')

    # Set memory limit consistently
    memory_limit = int(8192 * gpu_memory_limit / 100)  # Convert % to MB
    try:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
        )
        print(f'Set GPU memory limit to {memory_limit/1024:.1f}GB')
    except Exception as e:
        print(f'Error setting memory limit: {e}')
"

# Configure environment variables for GPU performance
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CUDNN_RESET_RND_GEN_STATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OPTIMIZE_FOR_RTX_4070=1

# Try to increase max_map_count for better performance with large memory footprints
echo 262144 > /proc/sys/vm/max_map_count 2>/dev/null || echo "Note: Could not set max_map_count (requires root)"

# Create necessary directories if they don't exist
echo "Creating directory structure..."
mkdir -p /app/EnhancedTrainingResults/MemoryLog
mkdir -p /app/EnhancedTrainingResults/SystemLog
mkdir -p /app/EnhancedTrainingResults/BacktestLog
mkdir -p /app/EnhancedTrainingResults/Trades
mkdir -p /app/EnhancedTrainingResults/TemperatureLog
mkdir -p /app/EnhancedTrainingResults/OptimizationResults

# Check if this script was called with arguments
if [ $# -gt 0 ]; then
    echo "Running with command arguments: $@"
    python3 "$@"
else
    # Handle the case when Docker CMD runs this script without args
    echo "Starting Enhanced Crypto Trading System with default options"
    python3 main.py --optimize-for-gpu
fi

# Store the exit code
EXIT_CODE=$?

# Final cleanup
echo "Cleaning up memory after execution..."
python3 -c "
import gc
import tensorflow as tf
try:
    tf.keras.backend.clear_session()
    gc.collect()
    print('Memory cleanup completed')
except Exception as e:
    print(f'Error during cleanup: {e}')
"

# Exit with the application's exit code
exit $EXIT_CODE