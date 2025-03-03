"""
Enhanced Memory Management Utilities (Refactored)

This module provides functions for monitoring and managing memory usage,
particularly for deep learning applications using TensorFlow with GPU acceleration.
Specifically optimized for RTX 4070 laptop GPU on Windows 11 / Dockerized / TensorFlow stack.
"""

import logging
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any

import psutil

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
logger = logging.getLogger("MemoryUtils")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

###############################################################################
# MEMORY & GPU CONSTANTS
###############################################################################

# System memory thresholds (GB)
DEFAULT_MEMORY_THRESHOLD_GB = 20.0
HIGH_MEMORY_THRESHOLD_GB = 24.0
CRITICAL_MEMORY_THRESHOLD_GB = 28.0

# Percentage thresholds for overall system memory usage
MEMORY_WARNING_PERCENT = 70
MEMORY_HIGH_PERCENT = 80
MEMORY_CRITICAL_PERCENT = 90

# GPU memory configuration
GPU_MEMORY_LIMIT_PERCENT = 65
RTX_4070_TOTAL_MEMORY_MB = 8192
GPU_RESERVED_MEMORY_MB = 256

# GPU temperature thresholds
GPU_WARNING_TEMP = 75
GPU_CRITICAL_TEMP = 82

# Monitoring intervals (seconds)
DEFAULT_MONITORING_INTERVAL = 30
HIGH_LOAD_MONITORING_INTERVAL = 10

# Component-specific thresholds (legacy; still kept for compatibility)
COMPONENT_THRESHOLDS = {
    "data_loading": 18.0,
    "feature_engineering": 22.0,
    "data_preparation": 20.0,
    "model_training": 24.0,
    "model_tuning": 26.0,
    "model_inference": 18.0,
    "backtest": 22.0,
    "optimization": 24.0,
    "general": 20.0,
    "pipeline": 22.0,
    "monitoring": 16.0
}

# Standardized component thresholds (GB)
MEMORY_THRESHOLDS = {
    "data_fetch": 20,
    "data_process": 20,
    "feature_engineering": 20,
    "data_preparation": 20,
    "model_training": 24,
    "model_prediction": 18,
    "backtest": 20,
    "trading": 18,
    "default": 20
}

# Log throttling configuration
LOG_THROTTLE = {
    'last_gpu_log': 0,
    'last_memory_log': 0,
    'log_interval': 60,
    'debug_mode': False
}

###############################################################################
# HELPER FUNCTIONS
###############################################################################


def _check_docker_memory_usage() -> Optional[float]:
    """
    Check Docker container memory usage if we're inside a Docker container.
    Returns a usage percentage if successful, or None if not running in Docker
    or if any error occurs.
    """
    try:
        if not (os.path.exists('/proc/self/cgroup') and any('docker' in line for line in open('/proc/self/cgroup'))):
            return None  # Not in Docker

        # cgroup v1
        if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                memory_limit = int(f.read().strip())
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                memory_usage = int(f.read().strip())

            return (memory_usage / memory_limit) * 100.0

        # cgroup v2
        elif os.path.exists('/sys/fs/cgroup/memory.max'):
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                memory_limit_str = f.read().strip()
            if memory_limit_str != 'max':
                memory_limit = int(memory_limit_str)
                with open('/sys/fs/cgroup/memory.current', 'r') as f:
                    memory_usage = int(f.read().strip())

                return (memory_usage / memory_limit) * 100.0

    except Exception as ex:
        logger.warning(f"Error checking Docker memory usage: {ex}")

    return None


def _force_cuda_reset():
    """
    Attempt to forcibly reset CUDA device memory using ctypes or fallback mechanisms.
    """
    try:
        import ctypes
        for lib_name in ['libcudart.so', 'libcudart.so.11.0', 'cudart64_110']:
            try:
                cudart = ctypes.CDLL(lib_name)
                if hasattr(cudart, 'cudaDeviceReset'):
                    cudart.cudaDeviceReset()
                    break
            except:
                continue
    except Exception as e:
        logger.warning(f"CUDA memory reset failed: {e}")


def is_memory_constrained() -> bool:
    """
    Detect if we're running in a memory-constrained environment (e.g., <16GB RAM or
    Docker-limited to <32GB). Adjusts some globals if so.
    """
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)

    if total_memory_gb < 16:
        return True

    # Check Docker cgroup memory limit
    try:
        if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                docker_limit = int(f.read().strip()) / (1024 ** 3)
            if docker_limit < total_memory_gb and docker_limit < 32:
                return True
    except:
        pass

    return False


# Adjust thresholds if memory constrained
if is_memory_constrained():
    logger.info("Detected memory-constrained environment, reducing thresholds by 20%.")
    DEFAULT_MEMORY_THRESHOLD_GB *= 0.8
    HIGH_MEMORY_THRESHOLD_GB *= 0.8
    CRITICAL_MEMORY_THRESHOLD_GB *= 0.8

    for key in COMPONENT_THRESHOLDS:
        COMPONENT_THRESHOLDS[key] *= 0.8

    GPU_MEMORY_LIMIT_PERCENT = 60


###############################################################################
# PUBLIC FUNCTIONS (Signatures preserved)
###############################################################################


def get_threshold_for_component(component: str) -> float:
    """
    Get standardized memory threshold for a component, falling back to 'default'
    if no match is found.
    """
    # Attempt exact match on MEMORY_THRESHOLDS first
    if component in MEMORY_THRESHOLDS:
        return MEMORY_THRESHOLDS[component]

    # Try base component name
    base_component = component.split('_')[0]
    if base_component in MEMORY_THRESHOLDS:
        return MEMORY_THRESHOLDS[base_component]

    # Fall back to 'default'
    return MEMORY_THRESHOLDS["default"]


def setup_tensorflow_optimization(gpu_memory_limit_pct: int = None) -> bool:
    """
    Configure GPU memory growth and other TensorFlow optimizations. Specifically
    optimized for an RTX 4070 Laptop GPU (8GB). Returns True if successful.
    """
    try:
        import tensorflow as tf

        if gpu_memory_limit_pct is None:
            gpu_memory_limit_pct = GPU_MEMORY_LIMIT_PERCENT
        gpu_memory_limit_pct = max(30, min(90, gpu_memory_limit_pct))

        # Basic environment settings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
        os.environ['TF_ENABLE_CUDA_MALLOC_ASYNC'] = '1'
        os.environ['TF_CUDA_VIRTUAL_MEMORY_ALLOCATION_GRANULARITY_FRAC'] = '0.75'
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            logger.info(f"Found {len(physical_devices)} GPU(s): {[d.name for d in physical_devices]}")

            # Try memory growth
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except RuntimeError as e:
                    logger.warning(f"Error enabling memory growth for {device}: {e}")

            # Calculate explicit memory limit
            usable_memory_mb = RTX_4070_TOTAL_MEMORY_MB - GPU_RESERVED_MEMORY_MB
            memory_limit = int(usable_memory_mb * gpu_memory_limit_pct / 100)

            try:
                tf.config.set_logical_device_configuration(
                    physical_devices[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"GPU memory capped at ~{memory_limit/1024:.1f}GB.")
            except RuntimeError as e:
                logger.warning(f"Logical device config failed: {e}")

            # Additional session config approach
            try:
                gpu_options = tf.compat.v1.GPUOptions(
                    per_process_gpu_memory_fraction=gpu_memory_limit_pct / 100,
                    allow_growth=True
                )
                session_conf = tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options,
                    intra_op_parallelism_threads=6,
                    inter_op_parallelism_threads=2
                )
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                logger.info("Applied session-level memory constraints.")
            except Exception as e:
                logger.warning(f"Failed applying session config: {e}")

        else:
            logger.info("No GPU detected or accessible. CPU-only mode.")

        # Mixed precision
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Enabled mixed precision (float16).")
        except Exception as e:
            logger.warning(f"Could not set mixed precision: {e}")

        # Set threading (somewhat arbitrary defaults)
        try:
            total_cores = os.cpu_count() or 16
            intra_threads = min(12, total_cores - 2)
            inter_threads = 2
            tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
            tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
            logger.info(f"Thread parallelism set: intra_op={intra_threads}, inter_op={inter_threads}")
        except Exception as e:
            logger.warning(f"Error setting TF threading: {e}")

        # Enable XLA JIT
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("Enabled XLA JIT compilation.")
        except Exception as e:
            logger.warning(f"Error enabling XLA JIT: {e}")

        # Enable TensorFloat-32
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            logger.info("Enabled TensorFloat-32 (TF32).")
        except Exception as e:
            logger.warning(f"Error enabling TF32: {e}")

        # Adaptive batch sizes if available
        if hasattr(tf.data, 'experimental') and hasattr(tf.data.experimental, 'enable_adaptive_batch_sizes'):
            try:
                tf.data.experimental.enable_adaptive_batch_sizes()
                logger.info("Enabled adaptive batch sizes for TensorFlow datasets.")
            except Exception as e:
                logger.warning(f"Failed enabling adaptive batch sizes: {e}")

        return True
    except ImportError:
        logger.warning("TensorFlow not installed or not found.")
        return False


def memory_watchdog(
    threshold_gb: Optional[float] = None,
    force_cleanup: bool = False,
    component: str = "general"
) -> bool:
    """
    Memory monitor that checks usage and triggers graduated cleanup levels.
    Returns True if a cleanup was performed.
    """
    try:
        memory_gb = log_memory_usage(component=component)

        # Default threshold for the specific component
        if threshold_gb is None:
            threshold_gb = get_threshold_for_component(component)

        # Handle external throttle request (e.g., GPU temperature critical)
        throttle_requested = False
        if os.path.exists("throttle_request"):
            logger.warning(f"Temperature throttling requested for {component}. Forcing cleanup.")
            try:
                with open("throttle_request", "r") as f:
                    info = f.read()
                    logger.info(f"Throttle info: {info}")
                os.remove("throttle_request")
            except (IOError, PermissionError) as e:
                logger.warning(f"Could not process throttle_request file: {e}")
            throttle_requested = True
            force_cleanup = True

        # Check Docker usage
        docker_usage_pct = _check_docker_memory_usage()
        if docker_usage_pct and docker_usage_pct > 80:
            logger.warning(f"High Docker memory usage: {docker_usage_pct:.1f}% => forcing cleanup.")
            force_cleanup = True

        system_mem = psutil.virtual_memory()
        memory_percent_used = system_mem.percent

        # Determine cleanup level
        cleanup_level = 0
        if force_cleanup:
            cleanup_level = 3
        elif memory_percent_used > MEMORY_CRITICAL_PERCENT or memory_gb > CRITICAL_MEMORY_THRESHOLD_GB:
            cleanup_level = 3
            logger.warning(f"[{component}] Critical system memory: {memory_gb:.2f}GB / {memory_percent_used:.1f}%.")
        elif memory_percent_used > MEMORY_HIGH_PERCENT or memory_gb > HIGH_MEMORY_THRESHOLD_GB:
            cleanup_level = 2
            logger.warning(f"[{component}] High system memory: {memory_gb:.2f}GB / {memory_percent_used:.1f}%.")
        elif memory_percent_used > MEMORY_WARNING_PERCENT or memory_gb > threshold_gb:
            cleanup_level = 1
            logger.warning(f"[{component}] Elevated system memory: {memory_gb:.2f}GB / {memory_percent_used:.1f}%.")

        if cleanup_level == 0:
            # No cleanup needed
            return False

        # Perform progressive cleanup
        if cleanup_level >= 1:
            # Light
            if not clear_memory(level=1):
                cleanup_level = max(cleanup_level, 2)

        if cleanup_level >= 2:
            # Standard
            if not clear_memory(level=2):
                cleanup_level = 3

        if cleanup_level >= 3:
            # Deep
            clear_memory(level=3, force_cuda_reset=throttle_requested)

        # Check memory post-cleanup
        new_memory_gb = log_memory_usage(component=f"{component}_post_cleanup")
        freed = memory_gb - new_memory_gb

        if freed > 0.1:
            logger.info(f"[{component}] Freed {freed:.2f}GB; usage is now {new_memory_gb:.2f}GB.")
        else:
            logger.warning(
                f"[{component}] Cleanup might have been ineffective. Before: {memory_gb:.2f}GB, After: {new_memory_gb:.2f}GB"
            )
            # Emergency fallback
            if cleanup_level == 3 and (new_memory_gb > CRITICAL_MEMORY_THRESHOLD_GB or system_mem.percent > MEMORY_CRITICAL_PERCENT):
                logger.critical("Emergency memory situation! Attempting last-resort cleanup.")
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect(0)
                    gc.collect(1)
                    gc.collect(2)
                    logger.critical("Consider restarting the Python process if memory remains critical.")
                except Exception as e:
                    logger.error(f"Emergency cleanup error: {e}")

        return freed > 0.1

    except Exception as e:
        logger.error(f"Error in memory_watchdog: {e}\n{traceback.format_exc()}")
        try:
            import gc
            gc.collect()
        except:
            pass
        return False


def clear_memory(level: int = 2, force_cuda_reset: bool = False) -> bool:
    """
    Clear Python and (optionally) GPU memory. Higher levels do more aggressive cleanup.
    Returns True if cleanup succeeded without error.
    """
    try:
        import gc
        # Level 1
        if level >= 1:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()

        # Level 2
        if level >= 2:
            try:
                import tensorflow as tf
                for i in range(3):
                    gc.collect(i)
                tf.keras.backend.clear_session()
                # Reset GPU memory stats if possible
                physical_devices = tf.config.list_physical_devices('GPU')
                for dev in physical_devices:
                    try:
                        tf.config.experimental.reset_memory_stats(dev)
                    except Exception:
                        pass
                dummy = tf.random.normal([1, 1])
                del dummy
                tf.keras.backend.clear_session()
            except ImportError:
                for i in range(3):
                    gc.collect(i)

        # Level 3
        if level >= 3:
            try:
                import tensorflow as tf
                physical_devices = tf.config.list_physical_devices('GPU')
                for dev in physical_devices:
                    try:
                        tf.config.experimental.reset_memory_stats(dev)
                    except:
                        pass
                tf.keras.backend.clear_session()

                for _ in range(3):
                    dummy = tf.random.normal([1, 1])
                    del dummy
                    gc.collect()

                try:
                    from tensorflow.python.framework import ops
                    ops.reset_default_graph()
                except:
                    pass

                if force_cuda_reset:
                    _force_cuda_reset()

                # Clear pandas/numpy overhead
                try:
                    import pandas as pd
                    pd.reset_option("^display.", silent=True)
                except ImportError:
                    pass
                try:
                    import numpy as np
                    np.set_printoptions(precision=4, suppress=True, threshold=10)
                except ImportError:
                    pass

                os.system('sync')
            except ImportError:
                for i in range(3):
                    gc.collect(i)

        return True
    except Exception as e:
        logger.error(f"Error in clear_memory: {e}\n{traceback.format_exc()}")
        return False


def log_memory_usage(
    log_file: str = "EnhancedTrainingResults/MemoryLog/memory_log.csv",
    component: str = "general"
) -> float:
    """
    Log process and system memory usage. Throttles logs to avoid excessive output.
    Returns current memory usage (in GB).
    """
    try:
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 ** 3)

        system_mem = psutil.virtual_memory()
        total_gb = system_mem.total / (1024 ** 3)
        used_gb = system_mem.used / (1024 ** 3)
        percent_used = system_mem.percent

        current_time = time.time()
        should_log_detail = (
            (current_time - LOG_THROTTLE['last_memory_log'] >= LOG_THROTTLE['log_interval'])
            or (percent_used > MEMORY_HIGH_PERCENT)
            or LOG_THROTTLE['debug_mode']
        )

        if should_log_detail:
            LOG_THROTTLE['last_memory_log'] = current_time
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Write header if necessary
            if not os.path.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("timestamp,component,memory_gb,total_gb,used_gb,percent_used\n")

            # Append usage
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()},{component},{memory_gb:.4f},{total_gb:.4f},{used_gb:.4f},{percent_used:.2f}\n")

        # Check GPU info
        gpu_info = get_gpu_info()
        if gpu_info:
            log_gpu_memory(gpu_info, component)

        return memory_gb
    except Exception as e:
        logger.error(f"Error in log_memory_usage: {e}")
        return 0.0


def log_gpu_memory(gpu_info: List[Dict[str, Any]], component: str = "general") -> None:
    """
    Log GPU memory usage to a CSV file with throttling to prevent spamming logs.
    """
    current_time = time.time()
    high_usage = any((gpu['memory_total'] > 0 and gpu['memory_used'] / gpu['memory_total'] > 0.7) for gpu in gpu_info)
    high_temp = any(gpu['temperature'] > GPU_WARNING_TEMP for gpu in gpu_info)

    should_log_gpu = (
        (current_time - LOG_THROTTLE['last_gpu_log'] >= LOG_THROTTLE['log_interval'])
        or high_usage or high_temp
        or LOG_THROTTLE['debug_mode']
    )
    if not should_log_gpu:
        return

    LOG_THROTTLE['last_gpu_log'] = current_time

    log_file = "EnhancedTrainingResults/MemoryLog/gpu_memory_log.csv"
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("timestamp,component,gpu_id,memory_used_mb,memory_total_mb,utilization_percent,temperature\n")

        with open(log_file, "a") as f:
            for gpu in gpu_info:
                f.write(
                    f"{datetime.now()},{component},{gpu['id']},{gpu['memory_used']},"
                    f"{gpu['memory_total']},{gpu['utilization']},{gpu['temperature']}\n"
                )
    except Exception as e:
        logger.error(f"Error logging GPU memory: {e}")


def get_gpu_info() -> list:
    """
    Gather GPU info via nvidia-smi, NVML, TensorFlow, or Numba (in that order).
    Returns a list of dicts describing each GPU.
    """
    try:
        # 1) Try nvidia-smi
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.sm',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        model_name = parts[1]
                        is_4070 = "4070" in model_name
                        power_draw = 0.0
                        clock_speed = 0.0
                        if len(parts) > 6 and parts[6].replace('.', '', 1).isdigit():
                            power_draw = float(parts[6])
                        if len(parts) > 7 and parts[7].replace('.', '', 1).isdigit():
                            clock_speed = float(parts[7])

                        gpu_info.append({
                            'id': parts[0],
                            'name': model_name,
                            'is_rtx_4070': is_4070,
                            'temperature': float(parts[2]) if parts[2].isdigit() else 0,
                            'utilization': float(parts[3]) if parts[3].isdigit() else 0,
                            'memory_used': int(float(parts[4])) if parts[4].replace('.', '', 1).isdigit() else 0,
                            'memory_total': int(float(parts[5])) if parts[5].replace('.', '', 1).isdigit() else 0,
                            'power_draw': power_draw,
                            'clock_speed': clock_speed
                        })
            return gpu_info
        except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
            pass

        # 2) Try NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_info = []
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used // (1024 * 1024)
                mem_total = mem_info.total // (1024 * 1024)
                is_4070 = "4070" in name

                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power_draw = 0.0
                try:
                    clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                except:
                    clock_speed = 0.0

                gpu_info.append({
                    'id': str(i),
                    'name': name,
                    'is_rtx_4070': is_4070,
                    'temperature': temp,
                    'utilization': float(util),
                    'memory_used': mem_used,
                    'memory_total': mem_total,
                    'power_draw': power_draw,
                    'clock_speed': clock_speed
                })

            pynvml.nvmlShutdown()
            return gpu_info
        except Exception:
            pass

        # 3) Try TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            gpu_info = []
            for i, gpu in enumerate(gpus):
                try:
                    mem_stats = tf.config.experimental.get_memory_info(gpu)
                    current_mem = mem_stats['current'] / (1024 ** 2)
                    peak_mem = mem_stats['peak'] / (1024 ** 2)
                    device_name = gpu.name
                    is_4070 = "4070" in device_name
                    gpu_info.append({
                        'id': str(i),
                        'name': device_name,
                        'is_rtx_4070': is_4070,
                        'temperature': 0,
                        'utilization': 0,
                        'memory_used': int(current_mem),
                        'memory_total': RTX_4070_TOTAL_MEMORY_MB if is_4070 else 8192,
                        'power_draw': 0,
                        'clock_speed': 0
                    })
                except:
                    gpu_info.append({
                        'id': str(i),
                        'name': str(gpu),
                        'is_rtx_4070': "4070" in str(gpu),
                        'temperature': 0,
                        'utilization': 0,
                        'memory_used': 0,
                        'memory_total': RTX_4070_TOTAL_MEMORY_MB,
                        'power_draw': 0,
                        'clock_speed': 0
                    })
            return gpu_info
        except Exception:
            pass

        # 4) Try Numba
        try:
            from numba import cuda
            device_count = len(cuda.gpus)
            if device_count > 0:
                return [{
                    'id': '0',
                    'name': 'Unknown CUDA Device',
                    'is_rtx_4070': False,
                    'temperature': 0,
                    'utilization': 0,
                    'memory_used': 0,
                    'memory_total': 8192,
                    'power_draw': 0,
                    'clock_speed': 0
                }]
        except:
            pass

        return []
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return []


def start_memory_monitoring(interval: int = None) -> threading.Thread:
    """
    Spawns a background thread that periodically logs memory usage and triggers
    cleanup if needed. Returns the Thread object.
    """
    if interval is None:
        interval = DEFAULT_MONITORING_INTERVAL

    def monitoring_thread():
        logger.info(f"Starting memory monitoring thread (interval={interval}s).")
        current_interval = interval
        consecutive_high = 0

        while True:
            try:
                memory_gb = log_memory_usage(component="monitor")
                system_mem = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=None)

                high_load_detected = (system_mem.percent > 80 or cpu_percent > 80)
                if high_load_detected:
                    consecutive_high += 1
                    if consecutive_high > 2 and current_interval > HIGH_LOAD_MONITORING_INTERVAL:
                        current_interval = HIGH_LOAD_MONITORING_INTERVAL
                        logger.info(f"High load: adjusting monitoring interval to {current_interval}s.")
                else:
                    consecutive_high = max(0, consecutive_high - 1)
                    if consecutive_high == 0 and current_interval < interval:
                        current_interval = interval
                        logger.info(f"Load normalized: reverting monitoring interval to {interval}s.")

                # Trigger cleanup if memory is too high
                if system_mem.percent > MEMORY_CRITICAL_PERCENT:
                    logger.warning("Critical system memory => forcing deep cleanup.")
                    memory_watchdog(force_cleanup=True, component="monitor")
                elif system_mem.percent > MEMORY_HIGH_PERCENT:
                    logger.warning("High system memory => standard cleanup.")
                    memory_watchdog(component="monitor")

                # GPU checks
                gpu_info = get_gpu_info()
                now = time.time()
                # Log or check critical GPU states more frequently if debug or high load
                if (now - LOG_THROTTLE['last_gpu_log'] >= LOG_THROTTLE['log_interval'] * 0.5) or LOG_THROTTLE['debug_mode']:
                    for gpu in gpu_info:
                        total_mem = gpu['memory_total']
                        used_mem = gpu['memory_used']
                        mem_pct = (used_mem / total_mem * 100) if total_mem > 0 else 0
                        if gpu['temperature'] > GPU_CRITICAL_TEMP:
                            logger.warning(f"Critical GPU temp: {gpu['temperature']}°C => throttle request.")
                            with open("throttle_request", "w") as f:
                                f.write(f"{datetime.now()}: GPU temp critical ({gpu['temperature']}°C)")
                            memory_watchdog(force_cleanup=True, component="gpu_temp_critical")
                        elif gpu['temperature'] > GPU_WARNING_TEMP:
                            logger.warning(f"High GPU temp: {gpu['temperature']}°C.")

                        if mem_pct > 90:
                            logger.warning(f"Critical GPU memory => deep cleanup.")
                            memory_watchdog(force_cleanup=True, component="gpu_mem_critical")
                        elif mem_pct > 80:
                            logger.warning(f"High GPU memory => standard cleanup.")
                            memory_watchdog(component="gpu_mem_high")

                    LOG_THROTTLE['last_gpu_log'] = now

                # Docker memory usage check
                docker_usage = _check_docker_memory_usage()
                if docker_usage and docker_usage > 85:
                    logger.warning(f"Docker memory usage critical ({docker_usage:.1f}%). Deep cleanup.")
                    memory_watchdog(force_cleanup=True, component="docker_mem_critical")
                elif docker_usage and docker_usage > 75:
                    logger.warning(f"Docker memory usage high ({docker_usage:.1f}%). Standard cleanup.")
                    memory_watchdog(component="docker_mem_high")

            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                # On error, sleep a bit more
                time.sleep(max(current_interval * 2, 30))
                current_interval = interval

            time.sleep(current_interval)

    t = threading.Thread(target=monitoring_thread, daemon=True)
    t.start()
    return t


def optimize_memory_for_dataframe(df, convert_floats: bool = True, convert_ints: bool = True) -> None:
    """
    Optimize memory usage of a pandas DataFrame in-place by downcasting.
    """
    try:
        import pandas as pd
        import numpy as np

        if df is None or df.empty:
            return

        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                continue
            if convert_floats and df[col].dtype == 'float64':
                df[col] = df[col].astype(np.float32)
            elif convert_ints and df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')

        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"DataFrame memory: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction).")

    except Exception as e:
        logger.warning(f"Error optimizing DataFrame memory: {e}")


def reduce_tensor_precision(tensors_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert high-precision tensors (np.float64 / tf.float64) to lower precision
    to reduce memory footprint.
    """
    try:
        import numpy as np
        import tensorflow as tf

        result = {}
        for key, tensor in tensors_dict.items():
            if isinstance(tensor, np.ndarray):
                if np.issubdtype(tensor.dtype, np.floating):
                    result[key] = tensor.astype(np.float32)
                elif (np.issubdtype(tensor.dtype, np.integer) and
                      tensor.dtype not in (np.int8, np.uint8)):
                    # Attempt int16 if it fits
                    if tensor.min() >= np.iinfo(np.int16).min and tensor.max() <= np.iinfo(np.int16).max:
                        result[key] = tensor.astype(np.int16)
                    else:
                        result[key] = tensor
                else:
                    result[key] = tensor
            elif hasattr(tensor, 'dtype') and "tensorflow" in str(type(tensor)):
                # TensorFlow Tensor
                dtype = tensor.dtype
                if dtype == tf.float64:
                    result[key] = tf.cast(tensor, tf.float32)
                elif dtype in (tf.int64, tf.int32):
                    # Attempt int16 cast if safe
                    try:
                        min_val = tf.reduce_min(tensor)
                        max_val = tf.reduce_max(tensor)
                        if min_val >= -32768 and max_val <= 32767:
                            result[key] = tf.cast(tensor, tf.int16)
                        else:
                            result[key] = tensor
                    except:
                        result[key] = tensor
                else:
                    result[key] = tensor
            else:
                result[key] = tensor
        return result
    except Exception as e:
        logger.warning(f"Error reducing tensor precision: {e}")
        return tensors_dict


def check_tf_memory_growth_enabled() -> bool:
    """
    Check whether TensorFlow memory growth has been enabled for all detected GPUs.
    """
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            logger.warning("No GPU devices found.")
            return False
        for device in physical_devices:
            if not tf.config.experimental.get_memory_growth(device):
                logger.warning(f"Memory growth not enabled for {device}.")
                return False
        return True
    except Exception as e:
        logger.error(f"Error checking TF memory growth: {e}")
        return False


def profile_memory_usage(func_to_profile, *args, **kwargs):
    """
    Run the given function and measure memory usage before/after. Logs the results.
    Returns (function_result, memory_diff_in_GB).
    """
    import gc
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 ** 3)
    start_time = time.time()

    result = func_to_profile(*args, **kwargs)
    execution_time = time.time() - start_time

    gc.collect()
    memory_after = process.memory_info().rss / (1024 ** 3)
    memory_diff = memory_after - memory_before

    logger.info(
        f"Memory usage for {func_to_profile.__name__}: "
        f"Before={memory_before:.2f}GB, After={memory_after:.2f}GB, "
        f"Diff={memory_diff:.2f}GB, Time={execution_time:.2f}s"
    )
    return result, memory_diff
