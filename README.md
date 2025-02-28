# Enhanced Crypto Trading System

A modular, ML-powered cryptocurrency trading system with feature engineering, deep learning models, and advanced risk management.

## System Architecture

The system is divided into six main modules:

1. **config.py** - Environment setup & global configurations
   - TensorFlow optimization
   - Memory management
   - System monitoring

2. **data_manager.py** - Data fetching and processing
   - Binance API integration
   - Multi-timeframe data handling (30m, 4h, daily)
   - Open interest and funding rates

3. **feature_engineering.py** - Feature creation and engineering
   - Technical indicators across timeframes
   - Market regime detection
   - Volume profile analysis
   - Swing detection

4. **data_preparation.py** - Data preparation for ML
   - Sequence building for time series
   - Train/validation splitting
   - Feature normalization
   - Label creation

5. **model_management.py** - ML models and training
   - Custom model architecture with transformer blocks
   - Hyperparameter optimization
   - Model ensemble training
   - Trading-specific metrics

6. **trading_logic.py** - Signal generation and backtesting
   - Advanced risk management
   - Dynamic entry/exit strategies
   - Multi-timeframe signal confirmation
   - Walk-forward backtesting

7. **main.py** - Main entry point and orchestration

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Binance account and API keys

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install tensorflow keras-tuner pandas numpy matplotlib binance-futures-connector sklearn scipy
   ```
3. Set your Binance API keys (in the code or as environment variables)

## Usage

### Basic Usage

Run the system with default settings:

```python
python main.py
```

This will:
1. Fetch Bitcoin USDT-futures data from Binance
2. Process and engineer features
3. Train a deep learning model
4. Perform walk-forward backtesting
5. Output trading results

### Configuration Options

You can modify key parameters in `main.py`:

- Change `backtest=True` to `backtest=False` for live trading mode
- Adjust risk parameters in `AdvancedRiskManager`
- Modify signal thresholds in `EnhancedSignalProducer`
- Change model parameters in `EnhancedCryptoModel`

### Multiple Configuration Testing

To compare different trading configurations:

```python
# In main.py, comment out single run and uncomment:
# run_system_with_multiple_configs()
```

## System Outputs

- Trading signals and performance metrics
- Detailed backtest logs in `EnhancedTrainingResults/`
- Performance comparison between configurations
- Model artifacts for reuse

## Memory and Temperature Monitoring

The system includes utilities to monitor:
- RAM usage to prevent out-of-memory errors
- GPU temperature to prevent overheating
- Automatic throttling when thresholds are exceeded

## License

This project is provided for educational purposes only. Use at your own risk.

## Disclaimer

Trading cryptocurrencies involves significant risk of loss. This software is for educational purposes only and should not be used for actual trading without extensive testing and customization.