# Memory Optimization for Crypto Trading System

This document outlines the memory optimization improvements made to the Enhanced Crypto Trading System to ensure it runs efficiently on the specified hardware.

## System Specifications

- **CPU**: Intel® Core™ Ultra 7 155H 16C/22T
- **GPU**: NVIDIA® GeForce® RTX4070 Max-Performance 8GB GDDR6
- **RAM**: 64GB DDR5 5600MHz
- **Storage**: 1TB M.2 SSD PCIe 4.0 x4
- **Environment**: Windows 11, Docker Container, CUDA 12.5, Python 3.11

## Key Optimization Components

### 1. Memory Management Utilities (`memory_utils.py`)

- **Memory Monitoring**: Continuous background monitoring of RAM and GPU memory
- **Cleanup Levels**: Three-tiered cleanup system (light, standard, aggressive)
- **GPU Optimization**: Configuration for RTX 4070 8GB GPU with memory growth limits
- **Mixed Precision**: Automatic float16 mixed precision for improved GPU performance
- **Temperature Monitoring**: GPU temperature tracking with automatic throttling
- **DataFrame Optimization**: Utilities to minimize pandas DataFrame memory usage

### 2. Feature Engineering Improvements (`feature_engineering_optimized.py`)

- **Strategic Memory Checkpoints**: Placed at critical points in feature calculation
- **Chunked Processing**: Option to process data in smaller chunks (configurable size)
- **Zero Matrix Prevention**: Detection and handling of constant columns
- **Adaptive Feature Selection**: Under memory pressure, selects only most important features
- **Data Type Optimization**: Automatic conversion of float64 to float32 when possible

### 3. Trading Logic Refactoring (`trading_logic_optimized.py`)

- **Modular Design**: Separated signal generation from trade execution
- **Memory-Efficient Backtesting**: Improved walk-forward testing with configurable window sizes
- **Direct-to-Disk Results**: Saves trades to disk instead of keeping everything in memory
- **Signal Producer Classes**: Better organization of market analysis components
- **Adaptive Parameters**: Scaling of risk parameters based on available memory

### 4. Enhanced Main Script (`main_optimized.py`)

- **Graceful Degradation**: Dynamically adjusts parameters based on available memory
- **Robust Error Handling**: Better error recovery and reporting
- **Enhanced Configuration**: Improved handling of user configuration with defaults
- **Memory-Aware Execution**: Adjusts workload based on system resources
- **Improved Logging**: More detailed logging of memory usage and system state

### 5. Startup Optimizations (`startup_optimized.sh`)

- **GPU Configuration**: Proper TensorFlow setup for RTX 4070 GPU
- **Environment Variables**: Sets key variables for optimal performance
- **Automatic Directory Creation**: Creates required directory structure
- **Post-Run Cleanup**: Final memory cleanup after execution

## How to Use the Optimized System

1. **Preparation**:
   ```bash
   # Extract the optimization files
   cp memory_utils_enhanced.py memory_utils.py
   cp feature_engineering_optimized.py feature_engineering.py
   cp trading_logic_optimized.py trading_logic.py
   cp main_optimized.py main.py
   chmod +x startup_optimized.sh
   ```

2. **Running with Docker**:
   ```bash
   # Use the optimized startup script
   docker run -it --gpus all -v $(pwd):/app crypto-trading:latest /app/startup_optimized.sh
   ```

3. **Monitoring Memory Usage**:
   ```bash
   # View memory logs
   cat EnhancedTrainingResults/MemoryLog/memory_log.csv
   
   # View GPU logs
   cat EnhancedTrainingResults/MemoryLog/gpu_log.csv
   ```

## Memory Optimization Testing

The system includes a dedicated memory optimization testing script (`test_memory_optimization.py`) that allows you to:

1. Test data fetching with memory profiling
2. Evaluate feature engineering memory efficiency
3. Analyze data preparation memory impact
4. Stress test the entire pipeline

```bash
# Run the memory test script
python3 test_memory_optimization.py

# Options:
# --live             Use live data instead of cached data
# --no-chunks        Process data without chunking
# --chunk-size INT   Size of chunks for processing (default: 1000)
# --subsample INT    Use subsample of data for preparation test
```

## Configuration Options

The system now supports additional memory-specific configuration parameters:

```json
{
  "system": {
    "memory_threshold_gb": 24,
    "gpu_memory_limit_pct": 65,
    "chunk_size": 1000,
    "use_chunking": true
  }
}
```

## Tips for Maximum Performance

1. **Use Chunked Processing**: Enable `use_chunking` for large datasets
2. **Adjust Sequence Length**: For large datasets, use shorter sequence lengths (16-24)
3. **Limit Training Epochs**: Use fewer epochs (5-10) for model training
4. **Ensemble Size**: Use single models instead of ensembles for memory efficiency
5. **Mixed Precision**: Keep mixed precision enabled for significant memory savings
6. **Monitor Temperature**: Watch GPU temperature logs to avoid thermal throttling
7. **Regular Testing**: Run the memory test script regularly to identify bottlenecks

This optimized system should run efficiently on your hardware configuration while maintaining the core functionality of the original system.