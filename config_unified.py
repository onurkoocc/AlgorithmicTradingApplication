"""Unified configuration system for the Enhanced Crypto Trading System"""
import json
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Config")


class TradingConfig:
    # Default configuration values (combined from both approaches)
    DEFAULT_CONFIG = {
        # Data parameters
        "data": {
            "csv_30m": "btc_data_30m.csv",
            "csv_4h": "btc_data_4h.csv",
            "csv_daily": "btc_data_daily.csv",
            "csv_oi": "btc_open_interest.csv",
            "csv_funding": "btc_funding_rates.csv"
        },
        "feature_engineering": {
            "feature_scaling": True
        },
        "data_preparation": {
            "sequence_length": 24,
            "horizon": 8,
            "train_ratio": 0.7
        },
        "model": {
            "project_name": "enhanced_crypto_model",
            "max_trials": 5,
            "tuner_type": "bayesian",
            "model_save_path": "best_enhanced_model.keras",
            "label_smoothing": 0.1,
            "ensemble_size": 1
        },
        "signal": {
            "confidence_threshold": 0.4,
            "strong_signal_threshold": 0.7,
            "atr_multiplier_sl": 1.5,
            "use_regime_filter": True,
            "use_volatility_filter": True
        },
        "risk": {
            "initial_capital": 10000.0,
            "max_risk_per_trade": 0.02,
            "max_correlated_exposure": 0.06,
            "volatility_scaling": True,
            "target_annual_vol": 0.2
        },
        "system": {
            "memory_threshold_gb": 24,
            "gpu_memory_limit_pct": 65,
            "chunk_size": 1000,
            "use_chunking": True
        },
        # Legacy parameters
        "lookback_30m_candles": 8000,
        "lookback_4h_candles": 1200,
        "lookback_daily_candles": 200
    }

    def __init__(self, config_file=None):
        """Initialize configuration with defaults and apply overrides"""
        # Start with default configuration
        self._config = self.DEFAULT_CONFIG.copy()

        # Apply file overrides if config file provided
        if config_file:
            self._apply_file_overrides(config_file)

        # Apply environment overrides
        self._apply_env_overrides()

        # Create required directories
        self._setup_directories()

    def __getattr__(self, name):
        """Allow dot notation access with fallback to dict lookup"""
        # First check in top-level config
        if name in self._config:
            return self._config[name]

        # Then check in nested sections
        for section in self._config:
            if isinstance(self._config[section], dict) and name in self._config[section]:
                return self._config[section][name]

        raise AttributeError(f"Configuration has no attribute '{name}'")

    def get(self, key, default=None):
        """Get config value with default fallback using dot notation or direct key"""
        try:
            # Try accessing via __getattr__
            return self.__getattr__(key)
        except AttributeError:
            return default

    def _apply_file_overrides(self, config_file):
        """Apply configuration from file"""
        if not os.path.exists(config_file):
            logger.warning(f"Configuration file not found: {config_file}")
            return

        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)

            # Apply overrides (handles nested dictionaries)
            self._update_nested_dict(self._config, file_config)
            logger.info(f"Applied configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")

    def _update_nested_dict(self, d, u):
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Example: CRYPTO_DATA_CSV_30M -> data.csv_30m
        for section, section_config in self._config.items():
            if isinstance(section_config, dict):
                for key in section_config:
                    env_key = f"CRYPTO_{section.upper()}_{key.upper()}"
                    if env_key in os.environ:
                        # Convert to appropriate type
                        if isinstance(section_config[key], bool):
                            self._config[section][key] = os.environ[env_key].lower() in ('true', 'yes', '1')
                        elif isinstance(section_config[key], int):
                            self._config[section][key] = int(os.environ[env_key])
                        elif isinstance(section_config[key], float):
                            self._config[section][key] = float(os.environ[env_key])
                        else:
                            self._config[section][key] = os.environ[env_key]

    def _setup_directories(self):
        """Create required directories"""
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


# Singleton instance
_CONFIG_INSTANCE = None


def get_config(config_file=None):
    """Get or create the global configuration instance"""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = TradingConfig(config_file)
    return _CONFIG_INSTANCE