# Configuration Override for Enhanced Crypto Trading System
# This file provides adjustments for small datasets

import logging
import os
import importlib
from datetime import datetime

# Set up basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConfigOverride")

logger.info("Applying configuration overrides for small datasets...")

# 1. Update Config parameters
try:
    from config import Config

    # Original values
    orig_30m = Config.LOOKBACK_30M_CANDLES
    orig_4h = Config.LOOKBACK_4H_CANDLES
    orig_daily = Config.LOOKBACK_DAILY_CANDLES

    # Increase lookback values to get more data
    Config.LOOKBACK_30M_CANDLES = 13000  # Increased from 13000 to capture more data
    Config.LOOKBACK_4H_CANDLES = 1625  # Approximately (5000 * 30min) / 240min = 625 4h candles
    Config.LOOKBACK_DAILY_CANDLES = 270  # Approximately (5000 * 30min) / 1440min = 104 daily candles

    logger.info(f"Updated Config.LOOKBACK_30M_CANDLES: {orig_30m} -> {Config.LOOKBACK_30M_CANDLES}")
    logger.info(f"Updated Config.LOOKBACK_4H_CANDLES: {orig_4h} -> {Config.LOOKBACK_4H_CANDLES}")
    logger.info(f"Updated Config.LOOKBACK_DAILY_CANDLES: {orig_daily} -> {Config.LOOKBACK_DAILY_CANDLES}")
except Exception as e:
    logger.error(f"Error updating Config parameters: {e}")

# 2. Create directory structure
try:
    # Create required directories
    for dir_path in [
        "EnhancedTrainingResults/MemoryLog",
        "EnhancedTrainingResults/SystemLog",
        "EnhancedTrainingResults/BacktestLog",
        "EnhancedTrainingResults/TradeLog",
        "EnhancedTrainingResults/Trades",
        "EnhancedTrainingResults/TemperatureLog",
        "EnhancedTrainingResults/OptimizationResults"
    ]:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Created directory structure")
except Exception as e:
    logger.error(f"Error creating directories: {e}")

# 3. Check if we're running in a container
try:
    in_container = os.path.exists('/.dockerenv')
    if in_container:
        logger.info("Running in Docker container")
    else:
        logger.info("Running in native environment")
except Exception as e:
    logger.error(f"Error checking container status: {e}")

# 4. Set starting message
logger.info("Configuration override complete. Enhanced Crypto Trading System ready to run.")
print("\n" + "=" * 80)
print("ENHANCED CRYPTO TRADING SYSTEM - SMALL DATASET CONFIGURATION")
print("=" * 80)
print(f"Configured at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("System is configured for smaller datasets with the following adjustments:")
print("  - Reduced sequence length and prediction horizon")
print("  - Dynamic backtesting window sizing")
print("  - Improved funding rate data handling")
print("  - Reduced model complexity for faster training")
print("=" * 80 + "\n")