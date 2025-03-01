#!/usr/bin/env python3
import time
import os
import psutil
import datetime
import subprocess

"""
Unified monitoring script that tracks both memory usage and GPU temperature/utilization
Replaces separate memory_monitor.py and temperature_monitor.py for efficiency
"""

# Create needed directories
log_dir = "EnhancedTrainingResults/MonitorLog"
os.makedirs(log_dir, exist_ok=True)

# Define log files
memory_log_file = f"{log_dir}/memory_log.csv"
gpu_log_file = f"{log_dir}/gpu_log.csv"

# Initialize log files with headers
with open(memory_log_file, "w") as f:
    f.write("timestamp,total_gb,used_gb,free_gb,process_gb,percent_used\n")

with open(gpu_log_file, "w") as f:
    f.write("timestamp,gpu_id,temperature,utilization,memory_used,memory_total\n")


def get_gpu_info():
    """Get GPU temperature, utilization and memory usage"""
    try:
        # Run nvidia-smi to get GPU information
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            universal_newlines=True
        )

        # Parse the output
        gpu_info = []
        for line in output.strip().split('\n'):
            if line:
                parts = line.split(', ')
                index = parts[0].strip()
                temp = float(parts[1].strip())
                util = float(parts[2].strip())
                mem_used = int(parts[3].strip())
                mem_total = int(parts[4].strip())

                gpu_info.append({
                    'id': index,
                    'temperature': temp,
                    'utilization': util,
                    'memory_used': mem_used,
                    'memory_total': mem_total
                })
        return gpu_info
    except (subprocess.SubprocessError, FileNotFoundError, IndexError) as e:
        print(f"Error getting GPU info: {e}")
        # If nvidia-smi fails or isn't available, return empty list
        return []


# Main monitoring loop
while True:
    try:
        # Current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # System memory monitoring
        system = psutil.virtual_memory()
        total_gb = system.total / (1024 ** 3)
        used_gb = system.used / (1024 ** 3)
        free_gb = system.available / (1024 ** 3)
        percent = system.percent

        # Process memory monitoring
        process = psutil.Process(os.getpid())
        process_gb = process.memory_info().rss / (1024 ** 3)

        # Log memory information
        with open(memory_log_file, "a") as f:
            f.write(f"{timestamp},{total_gb:.4f},{used_gb:.4f},{free_gb:.4f},{process_gb:.4f},{percent:.1f}\n")

        # GPU monitoring
        gpu_info = get_gpu_info()

        if not gpu_info:
            with open(gpu_log_file, "a") as f:
                f.write(f"{timestamp},NA,NA,NA,NA,NA\n")
        else:
            with open(gpu_log_file, "a") as f:
                for gpu in gpu_info:
                    f.write(
                        f"{timestamp},{gpu['id']},{gpu['temperature']},{gpu['utilization']},{gpu['memory_used']},{gpu['memory_total']}\n")

                    # Alert and throttle on high temperature
                    if gpu['temperature'] > 75:  # Warning threshold - RTX 4070 safe operating temp is <85°C
                        print(f"WARNING: GPU temperature high: {gpu['temperature']}°C")

                        # Force throttling if temperature is too high
                        if gpu['temperature'] > 82:  # Critical threshold
                            print(f"CRITICAL: GPU temperature too high! Forcing throttling at {gpu['temperature']}°C")
                            # Request throttling from main application
                            with open("throttle_request", "w") as tr:
                                tr.write(f"{timestamp}: Temperature critical at {gpu['temperature']}°C")

        # Check CPU temperature if available (many systems)
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = max([temp.current for temp in temps['coretemp']])
                if cpu_temp > 85:  # CPU warning threshold
                    print(f"WARNING: CPU temperature high: {cpu_temp}°C")
        except:
            pass  # Not all systems have temperature sensors accessible

        # Sleep for 15 seconds - good balance between monitoring frequency and resource usage
        time.sleep(15)
    except Exception as e:
        print(f"Monitoring error: {e}")
        time.sleep(60)  # Longer sleep on error