import logging
import time
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

# Import our optimized components
from config import setup_tensorflow_optimization, clear_memory, start_unified_monitor, \
    memory_watchdog, log_memory_usage
from data_manager import BitcoinData
from feature_engineering import EnhancedCryptoFeatureEngineer
from data_preparation import CryptoDataPreparer
from model_management import EnhancedCryptoModel
from trading_logic import AdvancedRiskManager, EnhancedSignalProducer, EnhancedStrategyBacktester


###############################################################################
# MAIN SYSTEM CLASS
###############################################################################
class EnhancedCryptoTradingSystem:
    def __init__(self, config_file=None):
        # Set up environment
        os.nice(10)  # Lower process priority
        setup_tensorflow_optimization()

        # Load configuration from file if provided
        self.config = self._load_config(config_file)

        # Data components with Binance integration
        self.data_fetcher = BitcoinData(
            csv_30m=self.config.get('data', {}).get('csv_30m', 'btc_data_30m.csv'),
            csv_4h=self.config.get('data', {}).get('csv_4h', 'btc_data_4h.csv'),
            csv_daily=self.config.get('data', {}).get('csv_daily', 'btc_data_daily.csv'),
            csv_oi=self.config.get('data', {}).get('csv_oi', 'btc_open_interest.csv'),
            csv_funding=self.config.get('data', {}).get('csv_funding', 'btc_funding_rates.csv')
        )

        # Feature engineering with enhanced features
        self.feature_engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=self.config.get('feature_engineering', {}).get('feature_scaling', True)
        )

        # Data preparation
        seq_length = self.config.get('data_preparation', {}).get('sequence_length', 48)
        horizon = self.config.get('data_preparation', {}).get('horizon', 16)

        self.data_preparer = CryptoDataPreparer(
            sequence_length=seq_length,
            horizon=horizon,
            normalize_method='zscore',
            price_column='close',
            train_ratio=self.config.get('data_preparation', {}).get('train_ratio', 0.7)
        )

        # Enhanced deep learning model
        self.model = EnhancedCryptoModel(
            project_name=self.config.get('model', {}).get('project_name', "enhanced_crypto_model"),
            max_trials=self.config.get('model', {}).get('max_trials', 20),
            tuner_type=self.config.get('model', {}).get('tuner_type', "bayesian"),
            model_save_path=self.config.get('model', {}).get('model_save_path', "best_enhanced_model.keras"),
            label_smoothing=self.config.get('model', {}).get('label_smoothing', 0.1),
            ensemble_size=self.config.get('model', {}).get('ensemble_size', 1)
        )

        # Advanced signal generation
        self.signal_producer = EnhancedSignalProducer(
            confidence_threshold=self.config.get('signal', {}).get('confidence_threshold', 0.4),
            strong_signal_threshold=self.config.get('signal', {}).get('strong_signal_threshold', 0.7),
            atr_multiplier_sl=self.config.get('signal', {}).get('atr_multiplier_sl', 1.5),
            use_regime_filter=self.config.get('signal', {}).get('use_regime_filter', True),
            use_volatility_filter=self.config.get('signal', {}).get('use_volatility_filter', True)
        )

        # Advanced risk management
        self.risk_manager = AdvancedRiskManager(
            initial_capital=self.config.get('risk', {}).get('initial_capital', 10000.0),
            max_risk_per_trade=self.config.get('risk', {}).get('max_risk_per_trade', 0.02),
            max_correlated_exposure=self.config.get('risk', {}).get('max_correlated_exposure', 0.06),
            volatility_scaling=self.config.get('risk', {}).get('volatility_scaling', True),
            target_annual_vol=self.config.get('risk', {}).get('target_annual_vol', 0.2)
        )

        # Logger setup
        self.logger = self._setup_logger()

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
                'sequence_length': 48,
                'horizon': 16,
                'train_ratio': 0.7
            },
            'model': {
                'project_name': "enhanced_crypto_model",
                'max_trials': 20,
                'tuner_type': "bayesian",
                'model_save_path': "best_enhanced_model.keras",
                'label_smoothing': 0.1,
                'ensemble_size': 1
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
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)

                # Deep merge user config into default config
                def deep_merge(source, destination):
                    for key, value in source.items():
                        if isinstance(value, dict):
                            # Get node or create empty dict
                            node = destination.setdefault(key, {})
                            deep_merge(value, node)
                        else:
                            destination[key] = value
                    return destination

                return deep_merge(user_config, default_config)
            except Exception as e:
                print(f"Error loading config file: {e}. Using defaults.")
                return default_config
        else:
            return default_config

    def _setup_logger(self):
        """Set up logging for the system"""
        logger = logging.getLogger("EnhancedCryptoSystem")
        logger.setLevel(logging.INFO)

        # Create directory if it doesn't exist
        os.makedirs("EnhancedTrainingResults/SystemLog", exist_ok=True)

        # Create handlers
        file_handler = logging.FileHandler(
            f"EnhancedTrainingResults/SystemLog/system_log_{datetime.now():%Y%m%d_%H%M%S}.log")
        file_handler.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)

        return logger

    def run_pipeline(self, backtest=True, optimization_mode=False):
        """Run the full trading system pipeline with optimization"""
        start_time = time.time()
        self.logger.info("Starting trading system pipeline")

        # Create directory structure
        os.makedirs("EnhancedTrainingResults/MonitorLog", exist_ok=True)
        os.makedirs("EnhancedTrainingResults/BacktestLog", exist_ok=True)
        os.makedirs("EnhancedTrainingResults/Trades", exist_ok=True)
        os.makedirs("EnhancedTrainingResults/OptimizationResults", exist_ok=True)

        # Initialize memory log
        with open("EnhancedTrainingResults/MonitorLog/memory_log.csv", "w") as f:
            f.write("timestamp,memory_gb\n")

        # Start unified monitoring service
        start_unified_monitor()

        try:
            # Step 1: Fetch 30-minute market data
            self.logger.info("Fetching 30-minute market data")
            df_30m = self.data_fetcher.fetch_30m_data(live=not backtest)

            # Check if we have enough data to proceed
            if len(df_30m) < 100:  # Minimum required for meaningful analysis
                self.logger.error(
                    f"Insufficient data: only {len(df_30m)} 30-minute candles available. Need at least 100.")
                return pd.DataFrame()  # Return empty DataFrame

            # Step 2: Derive 4-hour and daily data from 30-minute data
            self.logger.info("Deriving 4-hour and daily data")
            df_4h = self.data_fetcher.derive_4h_data(df_30m)
            df_daily = self.data_fetcher.derive_daily_data(df_30m)

            # Step 3: Fetch open interest and funding rate data
            self.logger.info("Fetching open interest data")
            df_oi = self.data_fetcher.fetch_open_interest(live=not backtest)

            self.logger.info("Fetching funding rate data")
            df_funding = self.data_fetcher.fetch_funding_rates(live=not backtest)

            # Log data shapes
            self.logger.info(
                f"Data shapes - 30m: {df_30m.shape}, 4H: {df_4h.shape}, "
                f"Daily: {df_daily.shape}, OI: {df_oi.shape}, Funding: {df_funding.shape}"
            )

            # Step 4: Feature engineering with open interest and funding rate data
            self.logger.info("Performing feature engineering")
            features_30m = self.feature_engineer.process_data_3way(df_30m, df_4h, df_daily, df_oi, df_funding)

            self.logger.info(f"Created feature set with shape: {features_30m.shape}")
            memory_watchdog(threshold_gb=40)

            # Clean up raw data to free memory
            del df_30m, df_4h, df_daily
            clear_memory()

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
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            return pd.DataFrame()
        finally:
            self.logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")
            log_memory_usage()

    def _run_live_trading(self, features_30m, df_oi, df_funding):
        """Run live trading with real-time data"""
        # Load trained model or ensemble
        if os.path.exists(self.model.model_save_path):
            self.logger.info("Loading trained model")
            self.model.load_best_model()
        elif os.path.exists(f"{self.model.model_save_path.replace('.keras', '')}_ensemble_0.keras"):
            self.logger.info("Loading ensemble models")
            self.model.load_ensemble()
        else:
            self.logger.error("No trained models found. Please run in backtest mode first.")
            return pd.DataFrame()

        # Prepare data for prediction
        X_test, y_test, df_test, _ = self.data_preparer.prepare_test_data(features_30m)

        if len(X_test) > 0:
            # Generate predictions
            if hasattr(self.model, 'predict_with_ensemble') and self.model.ensemble_size > 1:
                self.logger.info("Generating ensemble predictions")
                preds, uncertainties = self.model.predict_with_ensemble(X_test)
                latest_probs = preds[-1]
                latest_uncertainty = uncertainties[-1]
                self.logger.info(f"Prediction uncertainties: {latest_uncertainty}")
            else:
                self.logger.info("Generating model predictions")
                preds = self.model.predict_signals(X_test)
                latest_probs = preds[-1]

            # Get the most recent funding rate
            latest_funding_rate = 0
            if df_funding is not None and not df_funding.empty and 'fundingRate' in df_funding.columns:
                latest_funding_rate = df_funding['fundingRate'].iloc[-1] if len(df_funding) > 0 else 0

            # Get trading signal with OI and funding rate integration
            latest_signal = self.signal_producer.get_signal(
                latest_probs,
                features_30m,
                funding_df=df_funding if not df_funding.empty else None,
                oi_df=df_oi if not df_oi.empty else None
            )

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

                # Calculate position size considering funding rate
                quantity = self.risk_manager.calculate_position_size(
                    latest_signal,
                    entry_price,
                    stop_loss,
                    volatility_regime,
                    funding_rate=latest_funding_rate
                )

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
        else:
            self.logger.warning("Insufficient data for prediction")

        return pd.DataFrame()  # No tradable signal

    def _run_backtest(self, features_30m, df_oi, df_funding):
        """Run standard backtest with current parameters"""
        # Create backtester with optimal window sizing
        backtester = EnhancedStrategyBacktester(
            data_df=features_30m,
            oi_df=df_oi,
            funding_df=df_funding,
            preparer=self.data_preparer,
            modeler=self.model,
            signal_producer=self.signal_producer,
            risk_manager=self.risk_manager,
            train_window_size=min(int(len(features_30m) * 0.6), 5000),
            test_window_size=min(int(len(features_30m) * 0.3), 1000),
            fixed_cost=0.001,
            variable_cost=0.0005,
            slippage=0.0005
        )

        # Run walk-forward backtest
        results = backtester.walk_forward_backtest()

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
        """Run hyperparameter optimization with multiple configurations"""
        # Define configurations to test
        configurations = [
            {
                'name': 'funding_optimized',
                'model_params': {
                    'ensemble_size': 1,
                    'max_trials': 20
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
                    'max_trials': 20
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
            },
            {
                'name': 'conservative',
                'model_params': {
                    'ensemble_size': 1,
                    'max_trials': 20
                },
                'signal_params': {
                    'confidence_threshold': 0.6,
                    'use_regime_filter': True,
                    'atr_multiplier_sl': 1.2
                },
                'risk_params': {
                    'max_risk_per_trade': 0.015,
                    'volatility_scaling': True
                }
            }
        ]

        # Create results directory
        results_dir = f"EnhancedTrainingResults/OptimizationResults/optimization_{datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(results_dir, exist_ok=True)

        # Run each configuration
        all_results = {}

        for config in configurations:
            config_name = config['name']
            print(f"\nRunning configuration: {config_name}")

            # Apply configuration parameters
            self.model.ensemble_size = config['model_params'].get('ensemble_size', 1)
            self.model.max_trials = config['model_params'].get('max_trials', 20)

            self.signal_producer.confidence_threshold = config['signal_params'].get('confidence_threshold', 0.4)
            self.signal_producer.use_regime_filter = config['signal_params'].get('use_regime_filter', True)
            self.signal_producer.atr_multiplier_sl = config['signal_params'].get('atr_multiplier_sl', 1.5)

            self.risk_manager.max_risk_per_trade = config['risk_params'].get('max_risk_per_trade', 0.02)
            self.risk_manager.volatility_scaling = config['risk_params'].get('volatility_scaling', True)

            # Run backtest
            results = self._run_backtest(features_30m, df_oi, df_funding)

            # Save results
            results.to_csv(f"{results_dir}/{config_name}_results.csv")
            all_results[config_name] = results

            # Clear memory
            clear_memory()

        # Compare and visualize results
        self._compare_results(all_results, results_dir)

        return pd.DataFrame([{
            'configuration': k,
            'final_equity': v['final_equity'].iloc[-1] if 'final_equity' in v.columns and len(v) > 0 else 0
        } for k, v in all_results.items()])

    def _compare_results(self, all_results, results_dir):
        """Compare and visualize results from different configurations"""
        # Create summary dataframe
        summary = []

        for config_name, results in all_results.items():
            if not isinstance(results, pd.DataFrame) or results.empty or 'final_equity' not in results.columns:
                continue

            final_equity = results['final_equity'].iloc[-1]
            iterations = len(results)
            avg_return_per_iter = (final_equity / self.risk_manager.initial_capital - 1) / iterations * 100

            summary.append({
                'configuration': config_name,
                'final_equity': final_equity,
                'iterations': iterations,
                'avg_return_per_iteration': avg_return_per_iter
            })

        if not summary:
            self.logger.warning("No valid results to compare")
            return

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{results_dir}/summary_comparison.csv", index=False)

        # Create visualization
        plt.figure(figsize=(10, 6))

        for config_name, results in all_results.items():
            if not isinstance(results, pd.DataFrame) or results.empty or 'final_equity' not in results.columns:
                continue

            # Calculate equity curve
            equity_curve = [self.risk_manager.initial_capital]
            for i in range(len(results)):
                equity_curve.append(results['final_equity'].iloc[i])

            # Plot
            plt.plot(range(len(equity_curve)), equity_curve, label=config_name)

        plt.xlabel('Iteration')
        plt.ylabel('Equity')
        plt.title('Equity Curves by Configuration')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_dir}/equity_comparison.png")

        self.logger.info(f"\nResults comparison saved to {results_dir}")

        # Print out the winner
        if summary:
            best_config = max(summary, key=lambda x: x['final_equity'])
            self.logger.info(f"\n=== Best Configuration ===")
            self.logger.info(f"Configuration: {best_config['configuration']}")
            self.logger.info(f"Final Equity: ${best_config['final_equity']:.2f}")
            self.logger.info(
                f"Return: {best_config['final_equity'] / self.risk_manager.initial_capital * 100 - 100:.2f}%")
            self.logger.info(f"Iterations: {best_config['iterations']}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Crypto Trading System")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--api-key", type=str, help="Binance API key")
    parser.add_argument("--api-secret", type=str, help="Binance API secret")
    args = parser.parse_args()

    # Set API credentials from args or environment
    if args.api_key:
        os.environ["BINANCE_API_KEY"] = args.api_key
    if args.api_secret:
        os.environ["BINANCE_API_SECRET"] = args.api_secret

    # Create necessary directories
    os.makedirs("EnhancedTrainingResults/BacktestLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/MonitorLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/SystemLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/Trades", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/OptimizationResults", exist_ok=True)

    # Create and run the system
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