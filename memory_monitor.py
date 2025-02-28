#!/usr/bin/env python3
import time
import os
import psutil
import datetime

"""
Simple memory monitoring script that runs in background and 
logs system memory usage.
"""

log_dir = "EnhancedTrainingResults/MemoryLog"
os.makedirs(log_dir, exist_ok=True)

log_file = f"{log_dir}/detailed_memory_log.csv"

# Create or clear the log file
with open(log_file, "w") as f:
    f.write("timestamp,total_gb,used_gb,free_gb,process_gb,percent_used\n")

while True:
    try:
        # System memory
        system = psutil.virtual_memory()
        total_gb = system.total / (1024 ** 3)
        used_gb = system.used / (1024 ** 3)
        free_gb = system.available / (1024 ** 3)
        percent = system.percent

        # Current process memory
        process = psutil.Process(os.getpid())
        process_gb = process.memory_info().rss / (1024 ** 3)

        # Current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{total_gb:.4f},{used_gb:.4f},{free_gb:.4f},{process_gb:.4f},{percent:.1f}\n")

        # Sleep for 10 seconds
        time.sleep(10)
    except Exception as e:
        print(f"Memory monitoring error: {e}")
        time.sleep(30)  # Longer sleep on error