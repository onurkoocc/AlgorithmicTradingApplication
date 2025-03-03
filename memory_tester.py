#!/usr/bin/env python3
"""
Memory Optimization Test Script

This script tests the memory optimization improvements by running the enhanced
data processing and feature engineering modules with careful memory monitoring.
"""

import argparse
import logging
import os
import time
from datetime import datetime

from data_manager import BitcoinData
from memory_utils import (
    memory_watchdog,
    log_memory_usage,
    setup_tensorflow_optimization,
    start_memory_monitoring,
    optimize_memory_for_dataframe,
    profile_memory_usage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"memory_test_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MemoryTest")


def setup_environment():
    """Set up the environment for memory optimization"""
    logger.info("Setting up environment for memory optimization")

    # Create necessary directories
    os.makedirs("EnhancedTrainingResults/MemoryLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/SystemLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/BacktestLog", exist_ok=True)

    # Configure TensorFlow for optimal memory usage
    setup_tensorflow_optimization(gpu_memory_limit_pct=60)

    # Start memory monitoring in the background
    start_memory_monitoring(interval=15)

    # Lower process priority to avoid system freezes
    try:
        os.nice(10)
        logger.info("Lowered process priority")
    except (AttributeError, OSError):
        logger.warning("Could not lower process priority")


def fetch_data(use_cached=True):
    """Fetch data with memory monitoring"""
    logger.info("Fetching market data")

    # Initial memory checkpoint
    log_memory_usage(component="data_fetch_start")

    try:
        # Initialize data fetcher
        data_fetcher = BitcoinData(
            csv_30m='btc_data_30m.csv',
            csv_4h='btc_data_4h.csv',
            csv_daily='btc_data_daily.csv',
            csv_oi='btc_open_interest.csv',
            csv_funding='btc_funding_rates.csv'
        )

        # Fetch data with memory monitoring
        df_30m, fetch_time = profile_memory_usage(
            data_fetcher.fetch_30m_data,
            live=not use_cached
        )

        logger.info(f"Fetched 30m data: {len(df_30m)} rows, memory impact: {fetch_time:.2f}GB")

        # Memory checkpoint after 30m data
        memory_watchdog(threshold_gb=20, component="after_30m_fetch")

        # Derive 4h and daily data
        logger.info("Deriving 4h data")
        df_4h, derive_4h_mem = profile_memory_usage(
            data_fetcher.derive_4h_data,
            df_30m
        )

        logger.info(f"Derived 4h data: {len(df_4h)} rows, memory impact: {derive_4h_mem:.2f}GB")

        # Memory checkpoint after 4h data
        memory_watchdog(threshold_gb=20, component="after_4h_derive")

        logger.info("Deriving daily data")
        df_daily, derive_daily_mem = profile_memory_usage(
            data_fetcher.derive_daily_data,
            df_30m
        )

        logger.info(f"Derived daily data: {len(df_daily)} rows, memory impact: {derive_daily_mem:.2f}GB")

        # Memory checkpoint after daily data
        memory_watchdog(threshold_gb=20, component="after_daily_derive")

        # Fetch additional data
        logger.info("Fetching open interest data")
        df_oi, oi_mem = profile_memory_usage(
            data_fetcher.fetch_open_interest,
            live=not use_cached
        )

        logger.info(f"Fetched open interest data: {len(df_oi)} rows, memory impact: {oi_mem:.2f}GB")

        logger.info("Fetching funding rate data")
        df_funding, funding_mem = profile_memory_usage(
            data_fetcher.fetch_funding_rates,
            live=not use_cached
        )

        logger.info(f"Fetched funding rate data: {len(df_funding)} rows, memory impact: {funding_mem:.2f}GB")

        # Memory checkpoint after all data fetching
        memory_watchdog(threshold_gb=20, component="after_all_data_fetch")

        # Optimize dataframes for memory efficiency
        logger.info("Optimizing dataframes for memory efficiency")
        optimize_memory_for_dataframe(df_30m)
        optimize_memory_for_dataframe(df_4h)
        optimize_memory_for_dataframe(df_daily)
        optimize_memory_for_dataframe(df_oi)
        optimize_memory_for_dataframe(df_funding)

        # Log data shapes
        logger.info(
            f"Data shapes - 30m: {df_30m.shape}, 4H: {df_4h.shape}, "
            f"Daily: {df_daily.shape}, OI: {df_oi.shape}, Funding: {df_funding.shape}"
        )

        return df_30m, df_4h, df_daily, df_oi, df_funding

    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        return None, None, None, None, None


def process_features(self, df_30m, df_4h, df_daily, oi_df=None, funding_df=None, use_chunks=True, chunk_size=1000):
    """Single entry point for feature processing with chunking option"""
    # Use chunked or direct processing based on parameter
    if use_chunks and len(df_30m) > chunk_size:
        self.logger.info(f"Using chunked processing with size {chunk_size}")
        return self.process_data_in_chunks(df_30m, df_4h, df_daily, chunk_size, oi_df, funding_df)
    else:
        self.logger.info("Using direct processing")
        return self.process_data_3way(df_30m, df_4h, df_daily, oi_df, funding_df)


def test_data_preparation(features, subsample_size=None):
    """Test data preparation with memory optimization"""
    logger.info("Testing data preparation")

    # Initial memory checkpoint
    log_memory_usage(component="data_prep_start")

    try:
        # Import data preparation module
        from data_preparation import CryptoDataPreparer

        # Use a subsample if specified
        if subsample_size and len(features) > subsample_size:
            logger.info(f"Using subsample of {subsample_size} rows from {len(features)} total rows")
            sampled_features = features.sample(subsample_size, random_state=42)
        else:
            sampled_features = features

        # Create data preparer with smaller parameters for testing
        data_preparer = CryptoDataPreparer(
            sequence_length=24,  # Reduced from 48
            horizon=8,  # Reduced from 16
            normalize_method='zscore',
            price_column='close',
            train_ratio=0.7
        )

        # Prepare data with memory profiling
        start_time = time.time()
        prep_result, prep_mem = profile_memory_usage(
            data_preparer.prepare_data,
            sampled_features
        )

        # Unpack results if valid
        if isinstance(prep_result, tuple) and len(prep_result) >= 6:
            X_train, y_train, X_val, y_val, df_val, fwd_returns_val = prep_result

            processing_time = time.time() - start_time

            # Log results
            logger.info(f"Data preparation completed in {processing_time:.2f} seconds")
            logger.info(f"Memory impact: {prep_mem:.2f}GB")

            if X_train is not None:
                logger.info(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")

            if X_val is not None:
                logger.info(f"Validation data shape: X_val {X_val.shape}, y_val {y_val.shape}")

            # Final memory cleanup
            memory_watchdog(threshold_gb=20, component="after_data_preparation", force_cleanup=True)

            return True
        else:
            logger.warning("Data preparation did not return valid results")
            return False

    except Exception as e:
        logger.error(f"Error in data preparation: {e}", exc_info=True)
        return False


def run_memory_test(use_cached=True, use_chunked=True, chunk_size=1000, subsample_size=None):
    """Run the full memory test workflow"""
    logger.info("Starting memory optimization test")

    # Set up environment
    setup_environment()

    # Fetch data
    df_30m, df_4h, df_daily, df_oi, df_funding = fetch_data(use_cached=use_cached)

    if df_30m is None:
        logger.error("Data fetching failed, cannot continue")
        return False

    # Check if we have enough data
    if len(df_30m) < 200:
        logger.error(f"Insufficient data: only {len(df_30m)} 30-minute candles available. Need at least 200.")
        return False

    # Process features
    features, feature_stats = process_features(
        df_30m, df_4h, df_daily, df_oi, df_funding,
        use_chunked=use_chunked,
        chunk_size=chunk_size
    )

    if features is None:
        logger.error("Feature engineering failed, cannot continue")
        return False

    # Test data preparation
    prep_success = test_data_preparation(features, subsample_size=subsample_size)

    if not prep_success:
        logger.error("Data preparation test failed")
        return False

    logger.info("Memory optimization test completed successfully")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test memory optimization improvements")
    parser.add_argument("--live", action="store_true", help="Use live data instead of cached data")
    parser.add_argument("--no-chunks", action="store_true", help="Process data without chunking")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of chunks for processing")
    parser.add_argument("--subsample", type=int, default=None, help="Use subsample of data for preparation test")

    args = parser.parse_args()

    success = run_memory_test(
        use_cached=not args.live,
        use_chunked=not args.no_chunks,
        chunk_size=args.chunk_size,
        subsample_size=args.subsample
    )

    exit(0 if success else 1)