import datetime
import logging
import os
import threading
import time

import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("EnhancedTrainingResults/MonitorLog/unified_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedMonitor")

# Create directories
os.makedirs("EnhancedTrainingResults/MonitorLog", exist_ok=True)
os.makedirs("EnhancedTrainingResults/TemperatureLog", exist_ok=True)


class SystemMonitor:
    def __init__(self, interval=15):
        self.interval = interval
        self.running = False
        self.thread = None
        self.memory_log_path = "EnhancedTrainingResults/MonitorLog/memory_log.csv"
        self.gpu_log_path = "EnhancedTrainingResults/MonitorLog/gpu_log.csv"

        # Initialize log files with headers
        self._initialize_log_files()

    def _initialize_log_files(self):
        """Initialize log files with headers"""
        with open(self.memory_log_path, "w") as f:
            f.write("timestamp,total_gb,used_gb,free_gb,process_gb,percent_used\n")

        with open(self.gpu_log_path, "w") as f:
            f.write("timestamp,gpu_id,temperature,utilization,memory_used,memory_total\n")

    def get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        try:
            import subprocess
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                universal_newlines=True
            )

            # Parse the output
            gpu_info = []
            for line in output.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_info.append({
                            'id': parts[0],
                            'temperature': float(parts[1]) if parts[1].isdigit() else 0,
                            'utilization': float(parts[2]) if parts[2].isdigit() else 0,
                            'memory_used': int(parts[3]) if parts[3].isdigit() else 0,
                            'memory_total': int(parts[4]) if parts[4].isdigit() else 0
                        })
            return gpu_info
        except (subprocess.SubprocessError, FileNotFoundError, IndexError, ValueError) as e:
            logger.error(f"Error getting GPU info: {e}")
            return []

    def log_system_metrics(self):
        """Log system metrics to files"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # System memory metrics
        system = psutil.virtual_memory()
        total_gb = system.total / (1024 ** 3)
        used_gb = system.used / (1024 ** 3)
        free_gb = system.available / (1024 ** 3)
        percent = system.percent

        # Process memory
        process = psutil.Process(os.getpid())
        process_gb = process.memory_info().rss / (1024 ** 3)

        # Log memory information
        with open(self.memory_log_path, "a") as f:
            f.write(f"{timestamp},{total_gb:.4f},{used_gb:.4f},{free_gb:.4f},{process_gb:.4f},{percent:.1f}\n")

        # GPU monitoring
        gpu_info = self.get_gpu_info()

        with open(self.gpu_log_path, "a") as f:
            if not gpu_info:
                f.write(f"{timestamp},NA,NA,NA,NA,NA\n")
            else:
                for gpu in gpu_info:
                    f.write(f"{timestamp},{gpu['id']},{gpu['temperature']},{gpu['utilization']},"
                            f"{gpu['memory_used']},{gpu['memory_total']}\n")

                    # Alert and throttle on high temperature
                    if gpu['temperature'] > 75:  # Warning threshold
                        logger.warning(f"WARNING: GPU temperature high: {gpu['temperature']}°C")

                        # Force throttling if temperature is too high
                        if gpu['temperature'] > 82:  # Critical threshold
                            logger.warning(
                                f"CRITICAL: GPU temperature too high! Forcing throttling at {gpu['temperature']}°C")
                            # Request throttling from main application
                            with open("throttle_request", "w") as tr:
                                tr.write(f"{timestamp}: Temperature critical at {gpu['temperature']}°C")

    def monitoring_thread(self):
        """Background thread for continuous monitoring"""
        logger.info("Starting monitoring thread")
        while self.running:
            try:
                self.log_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(self.interval * 2)  # Wait longer on error

    def start(self):
        """Start the monitoring thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Monitoring already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self.monitoring_thread)
        self.thread.daemon = True  # Allow the thread to exit when the main program exits
        self.thread.start()
        logger.info("Monitoring service started")

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval * 2)
        logger.info("Monitoring service stopped")


# Singleton instance
_monitor = None


def start_monitoring():
    """Start the monitoring service (singleton pattern)"""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    _monitor.start()
    return _monitor


def stop_monitoring():
    """Stop the monitoring service"""
    global _monitor
    if _monitor is not None:
        _monitor.stop()


# For backwards compatibility with existing code
def main():
    """Main function when run directly"""
    start_monitoring()

    # Keep the script running for a while
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_monitoring()


if __name__ == "__main__":
    main()