#!/usr/bin/env python3
import time
import os
import datetime
import subprocess

"""
GPU temperature monitoring script that runs in the background and
logs GPU temperature periodically.
"""

log_dir = "EnhancedTrainingResults/TemperatureLog"
os.makedirs(log_dir, exist_ok=True)

log_file = f"{log_dir}/gpu_temperature_log.csv"

# Create or clear the log file
with open(log_file, "w") as f:
    f.write("timestamp,gpu_id,temperature,utilization\n")


def get_gpu_info():
    try:
        # Run nvidia-smi to get GPU information
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )

        # Parse the output
        gpu_info = []
        for line in output.strip().split('\n'):
            if line:
                index, temp, util = line.split(', ')
                gpu_info.append({
                    'id': index.strip(),
                    'temperature': float(temp.strip()),
                    'utilization': float(util.strip())
                })
        return gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        # If nvidia-smi fails or isn't available (no GPU), return empty list
        return []


while True:
    try:
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get GPU information
        gpu_info = get_gpu_info()

        # If no GPUs are found, log a placeholder
        if not gpu_info:
            with open(log_file, "a") as f:
                f.write(f"{timestamp},NA,NA,NA\n")
        else:
            # Log info for each GPU
            with open(log_file, "a") as f:
                for gpu in gpu_info:
                    f.write(f"{timestamp},{gpu['id']},{gpu['temperature']},{gpu['utilization']}\n")

        # Sleep for 30 seconds
        time.sleep(30)
    except Exception as e:
        print(f"Temperature monitoring error: {e}")
        time.sleep(60)  # Longer sleep on error