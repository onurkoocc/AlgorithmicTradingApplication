#!/usr/bin/env python3
"""
Enhanced Crypto Trading System - Memory Optimized Main Script

This script serves as the main entry point for the trading system with enhanced
memory management, error handling, and modular components.
"""

import argparse
import json
import logging
import os
import time
import traceback
from datetime import datetime

import pandas as pd
import psutil

import config_unified
from data_manager import BitcoinData
from data_preparation import CryptoDataPreparer
from feature_engineering import EnhancedCryptoFeatureEngineer
from memory_utils import (
    memory_watchdog,
    clear_memory,
    log_memory_usage,
    setup_tensorflow_optimization,
    start_memory_monitoring
)
from model_management import EnhancedCryptoModel
from trading_logic import AdvancedRiskManager, EnhancedSignalProducer, EnhancedStrategyBacktester


###############################################################################
# MAIN SYSTEM CLASS
###############################################################################
class EnhancedCryptoTradingSystem:
    def __init__(self, config_file=None):
        # Set up environment and optimize for Intel Core Ultra 7 155H & RTX 4070
        self._setup_environment()

        # Setup logger first so it's available throughout initialization
        self.logger = self._setup_logger()

        # Load configuration from file if provided
        self.config = self._load_config(config_file)

        # Initialize memory monitoring
        self._start_monitoring()

        # Initialize system components
        self._initialize_components()

    def _setup_environment(self):
        """Set up optimized environment for the trading system"""
        # Lower process priority to be a better system citizen
        try:
            os.nice(10)  # Lower process priority
        except (AttributeError, OSError):
            # Not supported on all platforms
            pass

        # Set up TensorFlow optimizations for RTX 4070 GPU
        config = config_unified.get_config()
        gpu_memory_limit_pct = config.get("system", {}).get("gpu_memory_limit_pct", 65)
        setup_tensorflow_optimization(gpu_memory_limit_pct=gpu_memory_limit_pct)  # Use 65% of GPU memory max

        # Create required directories
        for directory in [
            "EnhancedTrainingResults/MemoryLog",
            "EnhancedTrainingResults/SystemLog",
            "EnhancedTrainingResults/BacktestLog",
            "EnhancedTrainingResults/TradeLog",
            "EnhancedTrainingResults/Trades",
            "EnhancedTrainingResults/TemperatureLog",
            "EnhancedTrainingResults/OptimizationResults"
        ]:
            os.makedirs(directory, exist_ok=True)

    def _start_monitoring(self):
        """Start memory and system monitoring"""
        # Start memory monitoring in a background thread
        self.monitoring_thread = start_memory_monitoring(interval=15)
        self.logger.info("Started memory monitoring service")

        # Initial memory check
        log_memory_usage(component="system_init")

    def _setup_logger(self):
        """Set up the system logger"""
        logger = logging.getLogger("EnhancedCryptoSystem")
        logger.setLevel(logging.INFO)

        # Create directory if it doesn't exist
        os.makedirs("EnhancedTrainingResults/SystemLog", exist_ok=True)

        # Create log file with timestamp
        log_path = f"EnhancedTrainingResults/SystemLog/system_log_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _load_config(self, config_file):
        """Load configuration from JSON file if provided, otherwise use defaults"""
        default_config = {
            'data': {
                'csv_30m': 'btc_data_30m.csv',
                'csv_4h': 'btc_data_4h.csv',
                'csv_daily': 'btc_data_daily.csv',
                'csv_oi': 'btc_open_interest.csv',
                'csv_funding': 'btc_funding_rates.csv',
                'api_key': os.getenv("BINANCE_API_KEY", ""),
                'api_secret': os.getenv("BINANCE_API_SECRET", "")
            },
            'feature_engineering': {
                'feature_scaling': True
            },
            'data_preparation': {
                'sequence_length': 24,  # Reduced from 48
                'horizon': 8,  # Reduced from 16
                'train_ratio': 0.7
            },
            'model': {
                'project_name': "enhanced_crypto_model",
                'max_trials': 5,  # Reduced from 20
                'tuner_type': "bayesian",
                'model_save_path': "best_enhanced_model.keras",
                'label_smoothing': 0.1,
                'ensemble_size': 1  # Reduced from 3
            },
            'signal': {
                'confidence_threshold': 0.4,
                'strong_signal_threshold': 0.7,
                'atr_multiplier_sl': 1.5,
                'use_regime_filter': True,
                'use_volatility_filter': True
            },
            'risk': {
                'initial_capital': 10000.0,
                'max_risk_per_trade': 0.02,
                'max_correlated_exposure': 0.06,
                'volatility_scaling': True,
                'target_annual_vol': 0.2
            },
            'system': {
                'memory_threshold_gb': 24,
                'gpu_memory_limit_pct': 65,
                'chunk_size': 1000,  # For chunked data processing
                'use_chunking': True  # Whether to use chunked processing
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)

                # Recursive deep merge of user config into default config
                def deep_merge(source, destination):
                    for key, value in source.items():
                        if isinstance(value, dict):
                            # Get node or create empty dict
                            node = destination.setdefault(key, {})
                            deep_merge(value, node)
                        else:
                            destination[key] = value
                    return destination

                merged_config = deep_merge(user_config, default_config.copy())
                self.logger.info(f"Loaded configuration from {config_file}")
                return merged_config

            except Exception as e:
                self.logger.error(f"Error loading config file: {e}")
                self.logger.info("Using default configuration")
                return default_config
        else:
            self.logger.info("Using default configuration")
            return default_config

    def _initialize_components(self):
        """Initialize system components with optimized parameters"""
        # OPTIMIZATION: Check available memory and adjust parameters accordingly
        mem = psutil.virtual_memory()
        system_memory_gb = mem.total / (1024 ** 3)
        mem_factor = min(1.0, system_memory_gb / 32)  # Scale based on available memory

        self.logger.info(f"System has {system_memory_gb:.1f}GB RAM, scaling parameters to {mem_factor:.2f}")

        # Data fetcher
        self.data_fetcher = BitcoinData(
            csv_30m=self.config['data']['csv_30m'],
            csv_4h=self.config['data']['csv_4h'],
            csv_daily=self.config['data']['csv_daily'],
            csv_oi=self.config['data']['csv_oi'],
            csv_funding=self.config['data']['csv_funding']
        )

        # Feature engineering
        self.feature_engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=self.config['feature_engineering']['feature_scaling']
        )

        # Data preparation with optimized parameters
        seq_length = int(self.config['data_preparation']['sequence_length'] * mem_factor)
        seq_length = max(16, min(seq_length, 48))  # Ensure reasonable bounds

        horizon = int(self.config['data_preparation']['horizon'] * mem_factor)
        horizon = max(4, min(horizon, 16))  # Ensure reasonable bounds

        self.logger.info(f"Using sequence_length={seq_length}, horizon={horizon} based on available memory")

        self.data_preparer = CryptoDataPreparer(
            sequence_length=seq_length,
            horizon=horizon,
            normalize_method='zscore',
            price_column='close',
            train_ratio=self.config['data_preparation']['train_ratio']
        )

        # Model with optimized parameters
        max_trials = max(1, min(5, int(self.config['model']['max_trials'] * mem_factor)))
        ensemble_size = max(1, min(2, int(self.config['model']['ensemble_size'] * mem_factor)))

        self.logger.info(f"Using max_trials={max_trials}, ensemble_size={ensemble_size} based on available memory")

        self.model = EnhancedCryptoModel(
            project_name=self.config['model']['project_name'],
            max_trials=max_trials,
            tuner_type=self.config['model']['tuner_type'],
            model_save_path=self.config['model']['model_save_path'],
            label_smoothing=self.config['model']['label_smoothing'],
            ensemble_size=ensemble_size,
            use_mixed_precision=True  # Enable mixed precision for RTX 4070
        )

        # Signal producer
        self.signal_producer = EnhancedSignalProducer(
            confidence_threshold=self.config['signal']['confidence_threshold'],
            strong_signal_threshold=self.config['signal']['strong_signal_threshold'],
            atr_multiplier_sl=self.config['signal']['atr_multiplier_sl'],
            use_regime_filter=self.config['signal']['use_regime_filter'],
            use_volatility_filter=self.config['signal']['use_volatility_filter']
        )

        # Risk manager
        self.risk_manager = AdvancedRiskManager(
            initial_capital=self.config['risk']['initial_capital'],
            max_risk_per_trade=self.config['risk']['max_risk_per_trade'],
            max_correlated_exposure=self.config['risk']['max_correlated_exposure'],
            volatility_scaling=self.config['risk']['volatility_scaling'],
            target_annual_vol=self.config['risk']['target_annual_vol']
        )

    def run_pipeline(self, backtest=True, optimization_mode=False):
        """Run the full trading system pipeline with improved error handling and memory management"""
        start_time = time.time()
        self.logger.info("Starting trading system pipeline")

        # Initial memory checkpoint
        log_memory_usage(component="pipeline_start")

        try:
            # Step 1: Fetch 30-minute market data
            self.logger.info("Fetching 30-minute market data")
            try:
                df_30m = self.data_fetcher.fetch_30m_data(live=not backtest)
            except Exception as e:
                self.logger.error(f"Error fetching 30-minute data: {e}")
                return pd.DataFrame()

            # Check if we have enough data to proceed
            if len(df_30m) < 100:  # Minimum required for meaningful analysis
                self.logger.error(
                    f"Insufficient data: only {len(df_30m)} 30-minute candles available. Need at least 100.")
                return pd.DataFrame()  # Return empty DataFrame

            # Memory checkpoint after data fetching
            memory_watchdog(threshold_gb=20, component="after_30m_fetch")

            # Step 2: Derive 4-hour and daily data from 30-minute data
            self.logger.info("Deriving 4-hour and daily data")
            try:
                df_4h = self.data_fetcher.derive_4h_data(df_30m)
                df_daily = self.data_fetcher.derive_daily_data(df_30m)
            except Exception as e:
                self.logger.error(f"Error deriving timeframe data: {e}")
                return pd.DataFrame()

            # Memory checkpoint after deriving timeframes
            memory_watchdog(threshold_gb=20, component="after_deriving_timeframes")

            # Step 3: Fetch open interest and funding rate data
            self.logger.info("Fetching auxiliary data (open interest and funding rates)")
            try:
                df_oi = self.data_fetcher.fetch_open_interest(live=not backtest)
                df_funding = self.data_fetcher.fetch_funding_rates(live=not backtest)
            except Exception as e:
                self.logger.warning(f"Error fetching auxiliary data: {e}")
                # We can continue without this data
                df_oi = pd.DataFrame()
                df_funding = pd.DataFrame()

            # Log data shapes
            self.logger.info(
                f"Data shapes - 30m: {df_30m.shape}, 4H: {df_4h.shape}, "
                f"Daily: {df_daily.shape}, OI: {df_oi.shape}, Funding: {df_funding.shape}"
            )

            # Memory checkpoint before feature engineering
            memory_watchdog(threshold_gb=20, component="before_feature_engineering")

            # Step 4: Feature engineering with memory optimization
            self.logger.info("Performing feature engineering")

            # OPTIMIZATION: Check if we should use chunked processing
            use_chunking = self.config.get('system', {}).get('use_chunking', True)
            chunk_size = self.config.get('system', {}).get('chunk_size', 1000)

            # Get current memory usage to decide if chunking is required
            mem = psutil.virtual_memory()
            if mem.percent > 70 and not use_chunking:
                self.logger.warning(f"Memory usage high ({mem.percent}%), forcing chunked processing")
                use_chunking = True

            try:
                use_chunking = self.config.get("system", {}).get("use_chunking", True)
                chunk_size = self.config.get("system", {}).get("chunk_size", 1000)
                features_30m = self.feature_engineer.process_features(
                    df_30m, df_4h, df_daily, df_oi, df_funding,
                    use_chunks=use_chunking,
                    chunk_size=chunk_size
                )
            except Exception as e:
                self.logger.error(f"Error in feature engineering: {e}\n{traceback.format_exc()}")
                return pd.DataFrame()

            self.logger.info(f"Created feature set with shape: {features_30m.shape}")

            # Memory checkpoint after feature engineering
            memory_watchdog(threshold_gb=20, component="after_feature_engineering")

            # Clean up raw data to free memory
            self.logger.info("Cleaning up raw data to free memory")
            del df_30m, df_4h, df_daily
            clear_memory(level=2)

            if not backtest:
                # Live trading mode
                self.logger.info("Live trading mode activated")
                return self._run_live_trading(features_30m, df_oi, df_funding)

            # Backtest mode with optional optimization
            if optimization_mode:
                self.logger.info("Starting hyperparameter optimization")
                return self._run_optimization(features_30m, df_oi, df_funding)
            else:
                self.logger.info("Starting standard backtest")
                return self._run_backtest(features_30m, df_oi, df_funding)

        except Exception as e:
            self.logger.error(f"Unhandled exception in pipeline: {e}\n{traceback.format_exc()}")
            return pd.DataFrame()
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            log_memory_usage(component="pipeline_end")

    def _run_live_trading(self, features_30m, df_oi, df_funding):
        """Run live trading with real-time data and enhanced error handling"""
        self.logger.info("Preparing for live trading")

        # Load trained model or ensemble
        try:
            if os.path.exists(self.model.model_save_path):
                self.logger.info("Loading trained model")
                self.model.load_best_model()
            elif os.path.exists(f"{self.model.model_save_path.replace('.keras', '')}_ensemble_0.keras"):
                self.logger.info("Loading ensemble models")
                self.model.load_ensemble()
            else:
                self.logger.error("No trained models found. Please run in backtest mode first.")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return pd.DataFrame()

        # Prepare data for prediction
        try:
            self.logger.info("Preparing test data for prediction")
            X_test, y_test, df_test, _ = self.data_preparer.prepare_test_data(features_30m)
        except Exception as e:
            self.logger.error(f"Error preparing test data: {e}")
            return pd.DataFrame()

        if len(X_test) == 0:
            self.logger.warning("Insufficient data for prediction")
            return pd.DataFrame()

        # Generate predictions
        try:
            self.logger.info("Generating predictions")
            if hasattr(self.model, 'predict_with_ensemble') and self.model.ensemble_size > 1:
                self.logger.info("Using ensemble prediction")
                preds, uncertainties = self.model.predict_with_ensemble(X_test)
                latest_probs = preds[-1]
                latest_uncertainty = uncertainties[-1]
                self.logger.info(f"Prediction uncertainties: {latest_uncertainty}")
            else:
                self.logger.info("Using single model prediction")
                preds = self.model.predict_signals(X_test)
                latest_probs = preds[-1]
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return pd.DataFrame()

        # Get the most recent funding rate
        latest_funding_rate = 0
        if df_funding is not None and not df_funding.empty and 'fundingRate' in df_funding.columns:
            latest_funding_rate = df_funding['fundingRate'].iloc[-1] if len(df_funding) > 0 else 0

        # Get trading signal with OI and funding rate integration
        try:
            latest_signal = self.signal_producer.get_signal(
                latest_probs,
                features_30m,
                funding_df=df_funding if not df_funding.empty else None,
                oi_df=df_oi if not df_oi.empty else None
            )
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return pd.DataFrame()

        # Log the signal with OI and funding context
        self.logger.info(f"Live Trading Signal => {latest_signal}")

        # Calculate position size if signal is actionable
        if "Buy" in latest_signal['signal_type'] or "Sell" in latest_signal['signal_type']:
            direction = 'long' if "Buy" in latest_signal['signal_type'] else 'short'
            entry_price = features_30m['close'].iloc[-1]
            stop_loss = latest_signal['stop_loss']

            # Get volatility regime
            volatility_regime = features_30m['volatility_regime'].iloc[
                -1] if 'volatility_regime' in features_30m.columns else 0

            try:
                # Calculate position size considering funding rate
                quantity = self.risk_manager.calculate_position_size(
                    latest_signal,
                    entry_price,
                    stop_loss,
                    volatility_regime,
                    funding_rate=latest_funding_rate
                )
            except Exception as e:
                self.logger.error(f"Error calculating position size: {e}")
                quantity = 0

            self.logger.info(f"Recommended position size: {quantity}")

            # Return trading decision for execution
            return pd.DataFrame([{
                'timestamp': features_30m.index[-1],
                'signal_type': latest_signal['signal_type'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': latest_signal.get('take_profit',
                                                 entry_price * 1.03 if direction == 'long' else entry_price * 0.97),
                'quantity': quantity,
                'confidence': latest_signal.get('confidence', 0),
                'funding_rate': latest_funding_rate
            }])

        return pd.DataFrame()  # No tradable signal

    def _run_backtest(self, features_30m, df_oi, df_funding):
        """Run standard backtest with optimized parameters and memory management"""
        self.logger.info("Setting up backtester")

        # Memory check before backtesting
        memory_watchdog(threshold_gb=20, component="before_backtest")

        # OPTIMIZATION: Create backtester with memory-efficient window sizes
        # Adjust window sizes based on available memory and dataset size
        mem = psutil.virtual_memory()
        mem_factor = max(0.3, min(1.0, (100 - mem.percent) / 100))  # Scale factor based on free memory

        # Calculate window sizes
        max_window_size = int(len(features_30m) * 0.7)  # At most 70% of data for training
        train_window = min(int(max_window_size * mem_factor), 1200)  # Max 1200 candlesticks
        test_window = min(int(len(features_30m) * 0.15 * mem_factor), 400)  # Max 400 candlesticks

        self.logger.info(f"Using train_window={train_window}, test_window={test_window} for backtesting")

        # Initialize backtester
        try:
            backtester = EnhancedStrategyBacktester(
                data_df=features_30m,
                oi_df=df_oi,
                funding_df=df_funding,
                preparer=self.data_preparer,
                modeler=self.model,
                signal_producer=self.signal_producer,
                risk_manager=self.risk_manager,
                train_window_size=train_window,
                test_window_size=test_window,
                fixed_cost=0.001,
                variable_cost=0.0005,
                slippage=0.0005,
                walk_forward_steps=4  # Lower value to reduce memory requirements
            )
        except Exception as e:
            self.logger.error(f"Error initializing backtester: {e}")
            return pd.DataFrame()

        # Run walk-forward backtest
        try:
            results = backtester.walk_forward_backtest()
        except Exception as e:
            self.logger.error(f"Error during backtesting: {e}")
            return pd.DataFrame()

        # Memory cleanup after backtesting
        memory_watchdog(threshold_gb=20, component="after_backtest", force_cleanup=True)

        # Check if we got meaningful results
        if isinstance(results, pd.DataFrame) and not results.empty:
            self.logger.info("\n=== Backtest Results ===")
            self.logger.info(f"Results summary:\n{results.describe()}")

            # Add a more detailed summary if final_equity exists
            if 'final_equity' in results.columns and len(results) > 0:
                final_equity = results['final_equity'].iloc[-1]
                initial_capital = self.risk_manager.initial_capital
                profit = final_equity - initial_capital
                roi = profit / initial_capital * 100

                # Calculate annualized return
                days = (features_30m.index[-1] - features_30m.index[0]).days
                annualized_return = ((1 + roi / 100) ** (365 / max(days, 1)) - 1) * 100 if days > 0 else 0

                self.logger.info(f"\nTrading Performance Summary:")
                self.logger.info(f"Initial Capital: ${initial_capital:.2f}")
                self.logger.info(f"Final Equity: ${final_equity:.2f}")
                self.logger.info(f"Total Profit: ${profit:.2f}")
                self.logger.info(f"Return on Investment: {roi:.2f}%")
                self.logger.info(f"Annualized Return: {annualized_return:.2f}%")
                self.logger.info(f"Test Period: {days} days")
        else:
            self.logger.warning("Backtest did not produce valid results.")

        return results

    def _run_optimization(self, features_30m, df_oi, df_funding):
        """Run hyperparameter optimization with multiple configurations and memory efficiency"""
        self.logger.info("Setting up hyperparameter optimization")

        # Memory check before optimization
        memory_watchdog(threshold_gb=20, component="before_optimization")

        # Define configurations to test - simplified for memory efficiency
        # We'll just test 2 configurations instead of 3+ to save memory
        configurations = [
            {
                'name': 'funding_optimized',
                'model_params': {
                    'ensemble_size': 1,
                    'max_trials': 5  # Reduced from 20
                },
                'signal_params': {
                    'confidence_threshold': 0.4,
                    'use_regime_filter': True,
                    'atr_multiplier_sl': 1.5
                },
                'risk_params': {
                    'max_risk_per_trade': 0.02,
                    'volatility_scaling': True
                }
            },
            {
                'name': 'aggressive',
                'model_params': {
                    'ensemble_size': 1,
                    'max_trials': 5  # Reduced from 20
                },
                'signal_params': {
                    'confidence_threshold': 0.5,
                    'use_regime_filter': True,
                    'atr_multiplier_sl': 2.0
                },
                'risk_params': {
                    'max_risk_per_trade': 0.03,
                    'volatility_scaling': True
                }
            }
        ]

        # Create results directory
        results_dir = f"EnhancedTrainingResults/OptimizationResults/optimization_{datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(results_dir, exist_ok=True)

        # Store results directly to disk instead of all in memory
        all_results = {}

        # Run each configuration
        for config in configurations:
            config_name = config['name']
            self.logger.info(f"\nRunning configuration: {config_name}")

            # Memory checkpoint before configuration
            memory_watchdog(threshold_gb=20, component=f"before_config_{config_name}")

            # Apply configuration parameters
            self.model.ensemble_size = config['model_params'].get('ensemble_size', 1)
            self.model.max_trials = config['model_params'].get('max_trials', 5)

            self.signal_producer.confidence_threshold = config['signal_params'].get('confidence_threshold', 0.4)
            self.signal_producer.use_regime_filter = config['signal_params'].get('use_regime_filter', True)
            self.signal_producer.atr_multiplier_sl = config['signal_params'].get('atr_multiplier_sl', 1.5)

            self.risk_manager.max_risk_per_trade = config['risk_params'].get('max_risk_per_trade', 0.02)
            self.risk_manager.volatility_scaling = config['risk_params'].get('volatility_scaling', True)

            # Run backtest with error handling
            try:
                results = self._run_backtest(features_30m, df_oi, df_funding)
            except Exception as e:
                self.logger.error(f"Error in configuration {config_name}: {e}")
                results = pd.DataFrame()  # Empty DataFrame on error

            # Save results to disk
            if not results.empty:
                results_path = f"{results_dir}/{config_name}_results.csv"
                results.to_csv(results_path)
                self.logger.info(f"Saved {config_name} results to {results_path}")

                # Store in memory only the configuration name and final equity
                final_equity = results['final_equity'].iloc[-1] if 'final_equity' in results.columns and len(
                    results) > 0 else 0
                all_results[config_name] = final_equity
            else:
                self.logger.warning(f"No valid results for configuration: {config_name}")
                all_results[config_name] = 0

            # Clear memory between configurations
            clear_memory(level=3)
            memory_watchdog(threshold_gb=20, component=f"after_config_{config_name}", force_cleanup=True)

        # Write summary of results
        summary_path = f"{results_dir}/summary_results.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            self.logger.info(f"Saved optimization summary to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving optimization summary: {e}")

        # Final memory cleanup
        memory_watchdog(threshold_gb=20, component="after_optimization", force_cleanup=True)

        # Find best configuration
        if all_results:
            best_config = max(all_results.items(), key=lambda x: x[1])
            self.logger.info(f"\nBest configuration: {best_config[0]} with final equity: ${best_config[1]:.2f}")
        else:
            self.logger.warning("No valid results from any configuration")

        # Return summary as DataFrame
        return pd.DataFrame([{"configuration": k, "final_equity": v} for k, v in all_results.items()])


###############################################################################
# COMMAND-LINE INTERFACE
###############################################################################
def main():
    """Entry point for command-line execution"""
    parser = argparse.ArgumentParser(description="Enhanced Crypto Trading System")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--api-key", type=str, help="Binance API key")
    parser.add_argument("--api-secret", type=str, help="Binance API secret")
    parser.add_argument("--optimize-for-gpu", action="store_true", help="Apply memory optimizations for GPU")
    args = parser.parse_args()

    if args.api_key:
        os.environ["BINANCE_API_KEY"] = args.api_key
    if args.api_secret:
        os.environ["BINANCE_API_SECRET"] = args.api_secret

    # Set optimization flag for RTX 4070
    if args.optimize_for_gpu:
        os.environ["OPTIMIZE_FOR_RTX_4070"] = "1"
        print("Applying memory optimizations for RTX 4070 GPU")

    # Create and run the system
    try:
        system = EnhancedCryptoTradingSystem(config_file=args.config)

        if args.optimize:
            print("Running hyperparameter optimization...")
            results = system.run_pipeline(backtest=True, optimization_mode=True)
        elif args.live:
            print("Running in live trading mode...")
            results = system.run_pipeline(backtest=False)
        else:
            print("Running in backtest mode...")
            results = system.run_pipeline(backtest=True)

        if results is not None and isinstance(results, pd.DataFrame) and not results.empty:
            print("\nResults Summary:")
            print(results.to_string())

    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())