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
    # Optimized lookback periods balancing data needs and memory usage
    LOOKBACK_30M_CANDLES = 10000  # Reduced from 13000 to improve performance
    LOOKBACK_4H_CANDLES = 1500  # Derived from 30m data
    LOOKBACK_DAILY_CANDLES = 250  # Derived from 30m data


def setup_tensorflow_optimization():
    """Configure GPU memory growth and other TensorFlow optimizations"""
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # Increase memory limit to 6GB for RTX 4070 (out of 8GB)
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]
        )
        print(f"GPU memory limited to 6GB on {physical_devices[0].name}")

    # Enable mixed precision for RTX 4070
    try:
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')
        print("Enabled mixed precision (float16) for faster computation")
    except Exception as e:
        print(f"Warning: Could not set mixed precision: {e}. Continuing without it.")

    # Optimize thread count for Intel Core Ultra 7 155H (16C/22T)
    tf.config.threading.set_intra_op_parallelism_threads(14)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    # Enable XLA JIT compilation
    try:
        tf.config.optimizer.set_jit(True)
        print("Enabled XLA JIT compilation for faster training")
    except AttributeError:
        print("Warning: XLA JIT compilation not available. Continuing without it.")

    return True


def clear_memory():
    """Clear TensorFlow session and force garbage collection"""
    # Clear TensorFlow session
    clear_session()

    # Force garbage collection
    for i in range(3):
        gc.collect(i)

    return True


def start_unified_monitor():
    """Start unified monitoring in background"""
    subprocess.Popen(["python3", "unified_monitor.py"])
    return True


def memory_watchdog(threshold_gb=40, force_cleanup=False):
    """Monitor memory usage and clear if threshold exceeded or explicitly forced

    Increased threshold for your 64GB system while leaving enough memory for OS and other apps
    """
    memory_gb = log_memory_usage()

    # Check for throttle request due to high temperature
    if os.path.exists("throttle_request"):
        print("WARNING: Temperature throttling requested. Forcing memory cleanup...")
        os.remove("throttle_request")  # Remove the request file
        force_cleanup = True

    # Only clean up if memory usage is actually high or explicitly forced
    if force_cleanup or (memory_gb > 0.8 * threshold_gb and memory_gb > 2.0):
        print(
            f"WARNING: Memory usage ({memory_gb:.2f}GB) exceeded {0.8 * threshold_gb:.2f}GB threshold. Clearing memory...")
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