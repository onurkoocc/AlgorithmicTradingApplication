import logging
import os
import subprocess
import gc
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session


###############################################################################
# ENVIRONMENT SETUP & GLOBAL CONFIGURATIONS
###############################################################################
class Config:
    # We only need to specify lookback for 30m data now,
    # as others are derived from this
    LOOKBACK_30M_CANDLES = 13000

    # We'll keep these for compatibility with existing code
    # but they're derived from 30m data so we can adjust them as needed
    LOOKBACK_4H_CANDLES = 2000  # Approximately (7000 * 30min) / 240min = 875 4h candles
    LOOKBACK_DAILY_CANDLES = 300  # Approximately (7000 * 30min) / 1440min = 146 daily candles


def setup_tensorflow_optimization():
    """Configure GPU memory growth and other TensorFlow optimizations"""
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # Set memory limit to 5GB to avoid OOM errors
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=5120)]
        )
        print(f"GPU memory limited to 5GB on {physical_devices[0].name}")

    # Try to set mixed precision policy with appropriate error handling
    try:
        # Check if mixed_precision is available as a direct attribute
        if hasattr(tf.config, 'mixed_precision'):
            tf.config.mixed_precision.set_global_policy("mixed_float16")
        # Alternative approach in newer TensorFlow versions
        elif hasattr(tf, 'keras'):
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                set_global_policy('mixed_float16')
            except (ImportError, AttributeError):
                # If still not available, try another approach for newer TF versions
                try:
                    from tensorflow.keras import mixed_precision
                    mixed_precision.set_global_policy('mixed_float16')
                except (ImportError, AttributeError):
                    print("Warning: Could not set mixed precision policy. Continuing without it.")
        else:
            print("Warning: Mixed precision not available in this TensorFlow version. Continuing without it.")
    except Exception as e:
        print(f"Warning: Could not set mixed precision policy: {e}. Continuing without it.")

    # Set optimal thread count
    tf.config.threading.set_intra_op_parallelism_threads(14)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Enable XLA JIT compilation
    try:
        tf.config.optimizer.set_jit(True)
    except AttributeError:
        print("Warning: XLA JIT compilation not available in this TensorFlow version. Continuing without it.")

    return True


def clear_memory():
    """Clear TensorFlow session and force garbage collection"""
    # Clear TensorFlow session
    clear_session()

    # Force garbage collection
    for i in range(3):
        gc.collect(i)

    return True


def start_memory_monitor():
    """Start memory monitoring in background"""
    subprocess.Popen(["python3", "memory_monitor.py"])
    return True


def start_temperature_monitor():
    """Start temperature monitoring in background"""
    subprocess.Popen(["python3", "temperature_monitor.py"])
    return True


def memory_watchdog(threshold_gb=16, force_cleanup=False):
    """Monitor memory usage and clear if threshold exceeded or explicitly forced"""
    memory_gb = log_memory_usage()

    # Only clean up if memory usage is actually high or explicitly forced
    if force_cleanup or (memory_gb > 0.8 * threshold_gb and memory_gb > 2.0):
        print(f"WARNING: Memory usage ({memory_gb:.2f}GB) exceeded {0.8 * threshold_gb:.2f}GB threshold. Clearing memory...")
        clear_memory()
        return True

    return False


def log_memory_usage(log_file="EnhancedTrainingResults/MemoryLog/memory_log.csv"):
    """Log current memory usage to file"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, "a") as f:
            from datetime import datetime
            f.write(f"{datetime.now()},{memory_gb:.4f}\n")

        return memory_gb
    except Exception as e:
        print(f"Error logging memory usage: {e}")
        return 0