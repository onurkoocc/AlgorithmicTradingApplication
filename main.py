import logging
import time
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import our modular components
from config import setup_tensorflow_optimization, clear_memory, start_memory_monitor, start_temperature_monitor, \
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
    def __init__(self):
        # Set up environment
        os.nice(10)  # Lower process priority
        setup_tensorflow_optimization()

        # Update Config for more data
        from config import Config
        Config.LOOKBACK_30M_CANDLES = 5000  # Increased to get more historical data

        # Data components with Binance integration and OI/funding rate data
        self.data_fetcher = BitcoinData(
            csv_30m='btc_data_30m.csv',
            csv_4h='btc_data_4h.csv',
            csv_daily='btc_data_daily.csv',
            csv_oi='btc_open_interest.csv',
            csv_funding='btc_funding_rates.csv'
        )

        # Feature engineering with enhanced features
        self.feature_engineer = EnhancedCryptoFeatureEngineer(feature_scaling=True)

        # Data preparation - MODIFIED for smaller datasets
        self.data_preparer = CryptoDataPreparer(
            sequence_length=48,  # Reduced from 144 to 48 (1 day at 30-min intervals)
            horizon=16,          # Reduced from 48 to 16 (8 hours at 30-min intervals)
            normalize_method='zscore',
            price_column='close',
            train_ratio=0.7      # Adjusted for better use of limited data
        )

        # Enhanced deep learning model - MODIFIED for smaller datasets
        self.model = EnhancedCryptoModel(
            project_name="enhanced_crypto_model",
            max_trials=20,       # Reduced from 100 to 20
            tuner_type="bayesian",
            model_save_path="best_enhanced_model.keras",
            label_smoothing=0.1,
            ensemble_size=1      # Reduced from 3 to 1 for smaller datasets
        )

        # Advanced signal generation - MODIFIED for smaller datasets
        self.signal_producer = EnhancedSignalProducer(
            confidence_threshold=0.4,
            strong_signal_threshold=0.7,
            atr_multiplier_sl=1.5,
            use_regime_filter=False,  # Disabled for smaller datasets
            use_volatility_filter=False # Disabled for smaller datasets
        )

        # Advanced risk management
        self.risk_manager = AdvancedRiskManager(
            initial_capital=10000.0,
            max_risk_per_trade=0.02,
            max_correlated_exposure=0.06,
            volatility_scaling=True,
            target_annual_vol=0.2
        )

        # Logger setup
        self.logger = self._setup_logger()

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

    def run_pipeline(self, backtest=True):
        """Run the full trading system pipeline with Binance data and OI/funding rate features"""
        start_time = time.time()
        self.logger.info("Starting trading system pipeline")

        # Create memory log directory
        os.makedirs("EnhancedTrainingResults/MemoryLog", exist_ok=True)

        with open("EnhancedTrainingResults/MemoryLog/memory_log.csv", "w") as f:
            f.write("timestamp,memory_gb\n")

        # Start monitoring services
        start_memory_monitor()
        start_temperature_monitor()

        # Step 1: Fetch 30-minute market data from Binance
        self.logger.info("Fetching 30-minute market data from Binance")
        df_30m = self.data_fetcher.fetch_30m_data(live=not backtest)

        # Check if we have enough data to proceed
        if len(df_30m) < 200:  # Minimum required for meaningful analysis
            self.logger.error(f"Insufficient data: only {len(df_30m)} 30-minute candles available. Need at least 200.")
            return pd.DataFrame()  # Return empty DataFrame

        # Step 2: Derive 4-hour and daily data from 30-minute data
        self.logger.info("Deriving 4-hour and daily data from 30-minute data")
        df_4h = self.data_fetcher.derive_4h_data(df_30m)
        df_daily = self.data_fetcher.derive_daily_data(df_30m)

        # Step 3: Fetch open interest and funding rate data
        self.logger.info("Fetching open interest data from Binance")
        df_oi = self.data_fetcher.fetch_open_interest(live=not backtest)

        self.logger.info("Fetching funding rate data from Binance")
        df_funding = self.data_fetcher.fetch_funding_rates(live=not backtest)

        # Step 4: Fetch liquidation data for additional insights
        self.logger.info("Fetching liquidation data from Binance")
        df_liquidations = self.data_fetcher.fetch_liquidation_data(live=not backtest)

        self.logger.info(
            f"Data shapes - 30m: {df_30m.shape}, 4H: {df_4h.shape}, Daily: {df_daily.shape}, OI: {df_oi.shape}, Funding: {df_funding.shape}, Liquidations: {df_liquidations.shape if df_liquidations is not None else 'None'}")

        # Step 5: Feature engineering with open interest and funding rate data
        self.logger.info("Performing feature engineering")
        features_30m = self.feature_engineer.process_data_3way(df_30m, df_4h, df_daily)

        self.logger.info(f"Created feature set with shape: {features_30m.shape}")
        memory_watchdog(threshold_gb=15)

        # Clean up memory
        del df_30m, df_4h, df_daily, df_oi, df_funding
        clear_memory()

        if not backtest:
            # Live trading mode
            self.logger.info("Live trading mode activated")

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

                # Get trading signal with OI and funding rate integration
                latest_signal = self.signal_producer.get_signal(latest_probs, features_30m)

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

                    # Calculate position size
                    quantity = self.risk_manager.calculate_position_size(
                        latest_signal, entry_price, stop_loss, volatility_regime)

                    self.logger.info(f"Recommended position size: {quantity}")
            else:
                self.logger.warning("Insufficient data for prediction")

            self.logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")
            return pd.DataFrame()

        # Backtest mode
        self.logger.info("Starting backtest")

        # Create enhanced backtester with dynamic window sizing based on available data
        backtester = EnhancedStrategyBacktester(
            data_df=features_30m,
            preparer=self.data_preparer,
            modeler=self.model,
            signal_producer=self.signal_producer,
            risk_manager=self.risk_manager,
            train_window_size=min(int(len(features_30m) * 0.6), 5000),  # Dynamic sizing with upper limit
            test_window_size=min(int(len(features_30m) * 0.3), 1000),  # Dynamic sizing with upper limit
            fixed_cost=0.001,
            variable_cost=0.0005,
            slippage=0.0005
        )

        # Run walk-forward backtest
        results = backtester.walk_forward_backtest()

        # Check if we got meaningful results
        if isinstance(results, pd.DataFrame) and not results.empty:
            self.logger.info("\n=== Backtest Results ===")
            self.logger.info(f"Results summary:\n{results}")

            # Add a more detailed summary if final_equity exists
            if 'final_equity' in results.columns and len(results) > 0:
                final_equity = results['final_equity'].iloc[-1]
                initial_capital = self.risk_manager.initial_capital
                profit = final_equity - initial_capital
                roi = profit / initial_capital * 100

                self.logger.info(f"\nTrading Performance Summary:")
                self.logger.info(f"Initial Capital: ${initial_capital:.2f}")
                self.logger.info(f"Final Equity: ${final_equity:.2f}")
                self.logger.info(f"Total Profit: ${profit:.2f}")
                self.logger.info(f"Return on Investment: {roi:.2f}%")
        else:
            self.logger.warning("Backtest did not produce valid results.")

        self.logger.info(f"\nPipeline completed in {time.time() - start_time:.2f} seconds.")
        log_memory_usage()

        return results


# Function to run multiple configurations in parallel for comparison
def run_system_with_multiple_configs():
    """Run multiple configurations in parallel for comparison"""
    # Define configurations to test
    configurations = [
        {
            'name': 'baseline',
            'model_params': {
                'ensemble_size': 1,
                'max_trials': 50
            },
            'signal_params': {
                'confidence_threshold': 0.3,
                'use_regime_filter': False
            }
        },
        {
            'name': 'enhanced',
            'model_params': {
                'ensemble_size': 3,
                'max_trials': 100
            },
            'signal_params': {
                'confidence_threshold': 0.4,
                'use_regime_filter': True
            }
        },
        {
            'name': 'aggressive',
            'model_params': {
                'ensemble_size': 3,
                'max_trials': 100
            },
            'signal_params': {
                'confidence_threshold': 0.5,
                'use_regime_filter': True
            }
        }
    ]

    # Create results directory
    results_dir = f"EnhancedTrainingResults/OptimizationResults/optimization_results_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(results_dir, exist_ok=True)

    # Run each configuration
    all_results = {}

    for config in configurations:
        print(f"\nRunning configuration: {config['name']}")

        # Create system with this configuration
        system = EnhancedCryptoTradingSystem()

        # Apply configuration parameters
        system.model.ensemble_size = config['model_params']['ensemble_size']
        system.model.max_trials = config['model_params']['max_trials']
        system.signal_producer.confidence_threshold = config['signal_params']['confidence_threshold']
        system.signal_producer.use_regime_filter = config['signal_params']['use_regime_filter']

        # Run backtest
        results = system.run_pipeline(backtest=True)

        # Save results
        results.to_csv(f"{results_dir}/{config['name']}_results.csv")
        all_results[config['name']] = results

    # Compare and visualize results
    compare_results(all_results, results_dir)


def compare_results(all_results, results_dir):
    """Compare and visualize results from different configurations"""
    # Create summary dataframe
    summary = []

    for config_name, results in all_results.items():
        final_equity = results['final_equity'].iloc[-1]
        iterations = len(results)
        avg_return_per_iter = (final_equity / 10000.0 - 1) / iterations * 100

        summary.append({
            'configuration': config_name,
            'final_equity': final_equity,
            'iterations': iterations,
            'avg_return_per_iteration': avg_return_per_iter
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{results_dir}/summary_comparison.csv", index=False)

    # Create visualization
    plt.figure(figsize=(10, 6))

    for config_name, results in all_results.items():
        # Calculate equity curve
        equity_curve = [10000]
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

    print(f"\nResults comparison saved to {results_dir}")


if __name__ == "__main__":
    # Set up environment
    os.environ["BYBIT_API_KEY"] = os.getenv("BYBIT_API_KEY", "XW2qoCu1zlZdA8FEW98y7Md55ZtJ7fpaV1un6QkZErMeHIY7VXYAY5J6FonVTUdy")
    os.environ["BYBIT_API_SECRET"] = os.getenv("BYBIT_API_SECRET", "ivuwpI0yYRPRvSrex0IHGLcF4QP6jWTcUMCsWJ0DSQ3retcwOSTcmm9yzB1PFaP2")

    # Create necessary directories
    os.makedirs("EnhancedTrainingResults/BacktestLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/MemoryLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/SystemLog", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/Trades", exist_ok=True)
    os.makedirs("EnhancedTrainingResults/TemperatureLog", exist_ok=True)

    # Run the system - choose your run mode here:

    # Option 1: Run a single instance with default parameters
    system = EnhancedCryptoTradingSystem()
    results = system.run_pipeline(backtest=True)

    # Option 2: Run multiple configurations for comparison
    # run_system_with_multiple_configs()