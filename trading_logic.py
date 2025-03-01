import numpy as np
import pandas as pd
import os
import logging
import time
from datetime import datetime
from gc import collect
from scipy.stats import linregress
from sklearn.utils import compute_class_weight
from tensorflow.python.keras.backend import clear_session
from config import memory_watchdog, log_memory_usage

###############################################################################
# ADVANCED RISK MANAGEMENT
###############################################################################
class AdvancedRiskManager:
    def __init__(self, initial_capital=10000.0, max_risk_per_trade=0.02,
                 max_correlated_exposure=0.06, volatility_scaling=True,
                 target_annual_vol=0.2):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade  # As % of capital
        self.max_correlated_exposure = max_correlated_exposure  # Max risk for correlated positions
        self.volatility_scaling = volatility_scaling
        self.target_annual_vol = target_annual_vol
        self.open_positions = []
        self.trade_history = []
        self.performance_stats = {
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'drawdown': 0,
            'returns': []
        }

    # Update to AdvancedRiskManager's calculate_position_size method
    def calculate_position_size(self, signal, entry_price, stop_loss, volatility_regime=0, funding_rate=0):
        """
        Calculate optimal position size with funding rate and volatility considerations
        """
        if entry_price == stop_loss:
            return 0

        # Base risk amount as % of current capital
        base_risk_pct = self.max_risk_per_trade

        # Adjust risk based on recent performance
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-10:]
            win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)

            # Reduce risk after consecutive losses for drawdown protection
            consecutive_losses = 0
            for trade in reversed(recent_trades):
                if trade['pnl'] < 0:
                    consecutive_losses += 1
                else:
                    break

            # Implement Kelly-inspired position sizing based on win rate
            if consecutive_losses >= 3:
                base_risk_pct *= 0.5  # Cut risk by half after 3 consecutive losses
            elif win_rate < 0.4:
                base_risk_pct *= 0.7  # Reduce risk on low win rate
            elif win_rate > 0.6:
                base_risk_pct *= 1.2  # Increase risk on high win rate, but don't exceed maximum

        # Volatility scaling (critical for changing market conditions)
        if self.volatility_scaling:
            vol_factor = 1.0
            if volatility_regime > 0:  # High volatility - reduce size
                vol_factor = 0.7
            elif volatility_regime < 0:  # Low volatility - increase size
                vol_factor = 1.3
            base_risk_pct *= vol_factor

        # Funding rate adjustment - one of the key advantages in futures trading
        funding_factor = 1.0
        direction = 'long' if signal.get('signal_type', '').startswith('Buy') else 'short'

        # Funding rate creates a natural skew in expected returns:
        # - Negative funding: Shorts pay longs (favorable for longs)
        # - Positive funding: Longs pay shorts (favorable for shorts)
        if abs(funding_rate) > 0.0005:  # Significant funding rate
            if direction == 'long' and funding_rate < 0:
                # Negative funding rate favors longs (shorts pay longs)
                funding_factor = 1.2  # Increase size by 20%
            elif direction == 'short' and funding_rate > 0:
                # Positive funding rate favors shorts (longs pay shorts)
                funding_factor = 1.2  # Increase size by 20%
            elif direction == 'long' and funding_rate > 0.001:
                # High positive funding penalizes longs
                funding_factor = 0.8  # Decrease size by 20%
            elif direction == 'short' and funding_rate < -0.001:
                # High negative funding penalizes shorts
                funding_factor = 0.8  # Decrease size by 20%

        base_risk_pct *= funding_factor

        # Ensure we don't exceed max risk per trade
        risk_pct = min(base_risk_pct, self.max_risk_per_trade)

        # Calculate risk amount in currency
        risk_amount = self.current_capital * risk_pct

        # Calculate position size based on distance to stop loss
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            position_size = 0

        return position_size

    def check_correlation_risk(self, new_signal):
        """
        Check if adding this position would exceed maximum risk for correlated assets
        """
        # For crypto trading this is simpler since we're just trading one asset
        # but could be extended for multi-asset strategies

        current_exposure = sum(pos['risk_amount'] for pos in self.open_positions)
        current_exposure_pct = current_exposure / self.current_capital

        # If new position would cause total exposure to exceed limit, reject or scale down
        if current_exposure_pct + self.max_risk_per_trade > self.max_correlated_exposure:
            return False, self.max_correlated_exposure - current_exposure_pct

        return True, self.max_risk_per_trade

    def dynamic_exit_strategy(self, position, current_price, current_atr, funding_rate=0):
        """
        Implement significantly improved exit strategy with partial profits

        Key improvements:
        1. Tiered take-profit levels with progressive exits
        2. Dynamic trailing stop based on price movement
        3. Funding rate considerations for timing exits
        4. Volatility-adjusted stop distances
        """
        # Extract position details
        entry_price = position['entry_price']
        direction = position['direction']
        initial_stop = position['stop_loss']
        take_profit = position.get('take_profit', None)

        # Calculate current profit/loss
        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Determine trailing stop parameters based on ATR
        atr_multiple = 2.0  # Base ATR multiple for trailing stop

        # Rules for exit management
        if direction == 'long':
            # For long positions

            # Scenario 1: Small Profit (1%)
            # Take partial profit and keep position open
            if current_price >= entry_price * 1.01 and not position.get('partial_exit_1', False):
                return {
                    "partial_exit": True,
                    "exit_ratio": 0.25,  # Exit 25% of position
                    "reason": "FirstTarget",
                    "update_stop": True,
                    "new_stop": max(initial_stop, entry_price * 0.997)  # Move stop to -0.3% of entry
                }

            # Scenario 2: Medium Profit (2%)
            # Take more profit and tighten stop
            if current_price >= entry_price * 1.02 and not position.get('partial_exit_2', False):
                return {
                    "partial_exit": True,
                    "exit_ratio": 0.33,  # Exit 33% of remaining position
                    "reason": "SecondTarget",
                    "update_stop": True,
                    "new_stop": entry_price  # Move stop to breakeven
                }

            # Scenario 3: Good Profit (3%)
            # Take more profit and trail stop tightly
            if current_price >= entry_price * 1.03 and not position.get('partial_exit_3', False):
                return {
                    "partial_exit": True,
                    "exit_ratio": 0.5,  # Exit 50% of remaining position
                    "reason": "ThirdTarget",
                    "update_stop": True,
                    "new_stop": entry_price * 1.01  # Move stop to +1% profit
                }

            # Scenario 4: Excellent Profit (4%+)
            # Exit remaining position or trail very tightly
            if current_price >= entry_price * 1.04:
                # Check if funding rate has turned unfavorable for longs
                if funding_rate > 0.0001:  # Positive funding is unfavorable for longs
                    return {
                        "partial_exit": True,
                        "exit_ratio": 1.0,  # Exit all remaining
                        "reason": "FundingBasedExit"
                    }
                else:
                    # Otherwise just use a tight trail stop
                    new_stop = current_price * 0.99  # 1% trailing stop
                    if new_stop > initial_stop:
                        return {"update_stop": True, "new_stop": new_stop}

            # Default trailing logic based on ATR
            if pnl_pct > 0.005:  # If more than 0.5% in profit
                # Dynamic trailing factor - gets tighter as profit increases
                trail_factor = min(3.0, 1.0 + (pnl_pct * 10))
                new_stop = current_price - (atr_multiple / trail_factor * current_atr)

                # Only update if it would move the stop higher
                if new_stop > initial_stop:
                    return {"update_stop": True, "new_stop": new_stop}

        else:  # Short position logic
            # For short positions

            # Scenario 1: Small Profit (1%)
            # Take partial profit and keep position open
            if current_price <= entry_price * 0.99 and not position.get('partial_exit_1', False):
                return {
                    "partial_exit": True,
                    "exit_ratio": 0.25,  # Exit 25% of position
                    "reason": "FirstTarget",
                    "update_stop": True,
                    "new_stop": min(initial_stop, entry_price * 1.003)  # Move stop to +0.3% of entry
                }

            # Scenario 2: Medium Profit (2%)
            # Take more profit and tighten stop
            if current_price <= entry_price * 0.98 and not position.get('partial_exit_2', False):
                return {
                    "partial_exit": True,
                    "exit_ratio": 0.33,  # Exit 33% of remaining position
                    "reason": "SecondTarget",
                    "update_stop": True,
                    "new_stop": entry_price  # Move stop to breakeven
                }

            # Scenario 3: Good Profit (3%)
            # Take more profit and trail stop tightly
            if current_price <= entry_price * 0.97 and not position.get('partial_exit_3', False):
                return {
                    "partial_exit": True,
                    "exit_ratio": 0.5,  # Exit 50% of remaining position
                    "reason": "ThirdTarget",
                    "update_stop": True,
                    "new_stop": entry_price * 0.99  # Move stop to +1% profit
                }

            # Scenario 4: Excellent Profit (4%+)
            # Exit remaining position or trail very tightly
            if current_price <= entry_price * 0.96:
                # Check if funding rate has turned unfavorable for shorts
                if funding_rate < -0.0001:  # Negative funding is unfavorable for shorts
                    return {
                        "partial_exit": True,
                        "exit_ratio": 1.0,  # Exit all remaining
                        "reason": "FundingBasedExit"
                    }
                else:
                    # Otherwise just use a tight trail stop
                    new_stop = current_price * 1.01  # 1% trailing stop
                    if new_stop < initial_stop:
                        return {"update_stop": True, "new_stop": new_stop}

            # Default trailing logic based on ATR
            if pnl_pct > 0.005:  # If more than 0.5% in profit
                # Dynamic trailing factor - gets tighter as profit increases
                trail_factor = min(3.0, 1.0 + (pnl_pct * 10))
                new_stop = current_price + (atr_multiple / trail_factor * current_atr)

                # Only update if it would move the stop lower
                if new_stop < initial_stop:
                    return {"update_stop": True, "new_stop": new_stop}

        # No changes needed
        return {"update_stop": False}

    def update_performance_stats(self):
        """
        Update performance statistics based on trade history
        """
        if not self.trade_history:
            return

        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] < 0]

        self.performance_stats['win_rate'] = len(wins) / len(self.trade_history) if self.trade_history else 0
        self.performance_stats['avg_win'] = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        self.performance_stats['avg_loss'] = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        self.performance_stats['largest_win'] = max([t['pnl'] for t in wins]) if wins else 0
        self.performance_stats['largest_loss'] = min([t['pnl'] for t in losses]) if losses else 0

        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        self.performance_stats['profit_factor'] = total_profit / total_loss if total_loss else float('inf')

        # Calculate drawdown
        equity_curve = [self.initial_capital]
        for trade in self.trade_history:
            equity_curve.append(equity_curve[-1] + trade['pnl'])

        running_max = 0
        drawdowns = []

        for equity in equity_curve:
            if equity > running_max:
                running_max = equity
            drawdown = (running_max - equity) / running_max if running_max > 0 else 0
            drawdowns.append(drawdown)

        self.performance_stats['drawdown'] = max(drawdowns)

        # Calculate returns for Sharpe ratio
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

        self.performance_stats['returns'] = returns
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            self.performance_stats['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0

        return self.performance_stats


###############################################################################
# ENHANCED SIGNAL GENERATOR
###############################################################################
class EnhancedSignalProducer:
    def __init__(self, confidence_threshold=0.4, strong_signal_threshold=0.7,
                 atr_multiplier_sl=1.5, use_regime_filter=True, use_volatility_filter=True):
        self.confidence_threshold = confidence_threshold
        self.strong_signal_threshold = strong_signal_threshold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.min_adx_threshold = 20
        self.max_vol_percentile = 85  # Avoid trading in extremely high volatility
        self.correlation_threshold = 0.6  # For timeframe agreement

    def get_signal(self, model_probs, df, funding_df=None, oi_df=None):
        """
        Enhanced signal generation integrating funding rates and open interest

        This is one of the most critical components for profitability in crypto futures trading
        """
        if len(df) < 2:
            return {"signal_type": "NoTrade", "reason": "InsufficientData"}

        # Get base probabilities from model
        P_positive = model_probs[3] + model_probs[4]  # Classes 3 & 4 = bullish
        P_negative = model_probs[0] + model_probs[1]  # Classes 0 & 1 = bearish
        P_neutral = model_probs[2]  # Class 2 = neutral
        max_confidence = max(P_positive, P_negative)

        # Get current market conditions
        current_price = df['close'].iloc[-1]

        # Get relevant regime indicators if available
        market_regime = df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 0
        volatility_regime = df['volatility_regime'].iloc[-1] if 'volatility_regime' in df.columns else 0
        trend_strength = df['trend_strength'].iloc[-1] if 'trend_strength' in df.columns else 0

        # Check funding rate features if available (critical for futures)
        funding_signal = 0
        if 'funding_extreme_signal' in df.columns:
            funding_signal = df['funding_extreme_signal'].iloc[-1]
        elif 'funding_rate' in df.columns:
            funding_rate = df['funding_rate'].iloc[-1]
            if funding_rate > 0.0001:  # Positive funding (bearish)
                funding_signal = -1
            elif funding_rate < -0.0001:  # Negative funding (bullish)
                funding_signal = 1

        # Check open interest features if available
        oi_signal = 0
        if 'oi_price_sentiment' in df.columns:
            oi_signal = df['oi_price_sentiment'].iloc[-1]

        # Check if ATR data exists for volatility-based stops
        if 'd1_ATR_14' not in df.columns:
            # Fallback method to calculate ATR if not available
            atr = self._compute_atr(df).iloc[-1]
        else:
            atr = df['d1_ATR_14'].iloc[-1]

        # Calculate recent volatility measure
        hist_vol = df['hist_vol_20'].iloc[-1] if 'hist_vol_20' in df.columns else df['close'].pct_change(20).std()

        # Combined signal weighting
        # Base signal from model (60% weight)
        base_signal = 1 if P_positive > P_negative else (-1 if P_negative > P_positive else 0)
        combined_signal = base_signal * 0.6

        # External signals (40% total weight)
        if abs(funding_signal) > 0:
            combined_signal += funding_signal * 0.2  # Funding signal (20%)

        if abs(market_regime) > 0:
            combined_signal += market_regime * 0.1  # Market regime (10%)

        if abs(oi_signal) > 0:
            combined_signal += oi_signal * 0.1  # Open interest signal (10%)

        # Filter 1: Volatility Filter - avoid extremely high volatility periods
        if self.use_volatility_filter:
            if volatility_regime > 0 and hist_vol > np.percentile(df['hist_vol_20'].dropna(), self.max_vol_percentile):
                return {"signal_type": "NoTrade", "reason": "ExtremeVolatility"}

        # Filter 2: Confidence threshold - require minimum confidence
        if max_confidence < self.confidence_threshold:
            return {"signal_type": "NoTrade", "confidence": max_confidence, "reason": "LowConfidence"}

        # Filter 3: Trend alignment
        if self.use_regime_filter and abs(trend_strength) > 0.5:
            trend_aligned = (trend_strength > 0 and combined_signal > 0) or (trend_strength < 0 and combined_signal < 0)
            if not trend_aligned:
                return {"signal_type": "NoTrade", "reason": "TrendMisalignment"}

        # Generate trading signals based on combined analysis
        if combined_signal > 0.2:  # Bullish threshold
            # For long signals
            # Dynamic stop loss based on volatility
            volatility_factor = 1.0 + (0.5 * volatility_regime)  # Increase for higher volatility
            stop_loss_price = current_price - (self.atr_multiplier_sl * atr * volatility_factor)

            # Calculate take profit levels based on volatility
            tp_ratio = 2.5  # 2.5:1 reward-to-risk by default

            # Adjust ratio based on funding rate conditions
            if funding_signal > 0:  # Funding is favorable for longs
                tp_ratio = 3.0  # Increase reward-to-risk when funding is favorable

            take_profit_price = current_price + (tp_ratio * (current_price - stop_loss_price))

            # Determine signal strength
            if P_positive >= self.strong_signal_threshold:
                signal_str = "StrongBuy"
            else:
                signal_str = "Buy"

            return {
                "signal_type": signal_str,
                "confidence": float(P_positive),
                "stop_loss": round(float(stop_loss_price), 2),
                "take_profit": round(float(take_profit_price), 2),
                "regime": int(market_regime),
                "volatility": float(hist_vol),
                "funding_signal": funding_signal,
                "combined_signal": combined_signal
            }

        elif combined_signal < -0.2:  # Bearish threshold
            # For short signals
            volatility_factor = 1.0 + (0.5 * volatility_regime)  # Increase for higher volatility
            stop_loss_price = current_price + (self.atr_multiplier_sl * atr * volatility_factor)

            # Calculate take profit levels based on volatility
            tp_ratio = 2.5  # 2.5:1 reward-to-risk by default

            # Adjust ratio based on funding rate conditions
            if funding_signal < 0:  # Funding is favorable for shorts
                tp_ratio = 3.0  # Increase reward-to-risk when funding is favorable

            take_profit_price = current_price - (tp_ratio * (stop_loss_price - current_price))

            # Determine signal strength
            if P_negative >= self.strong_signal_threshold:
                signal_str = "StrongSell"
            else:
                signal_str = "Sell"

            return {
                "signal_type": signal_str,
                "confidence": float(P_negative),
                "stop_loss": round(float(stop_loss_price), 2),
                "take_profit": round(float(take_profit_price), 2),
                "regime": int(market_regime),
                "volatility": float(hist_vol),
                "funding_signal": funding_signal,
                "combined_signal": combined_signal
            }

        return {"signal_type": "NoTrade", "reason": "InsufficientSignal"}

    def _compute_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _detect_divergence(self, df):
        """Detect price-indicator divergence for additional signals"""
        if 'h4_RSI_14' not in df.columns or len(df) < 20:
            return False, False

        # Look for last 5 candles
        price_highs = df['high'].iloc[-20:].values
        price_lows = df['low'].iloc[-20:].values
        rsi_values = df['h4_RSI_14'].iloc[-20:].values

        # Find local extremes
        price_high_idx = np.argmax(price_highs)
        price_low_idx = np.argmin(price_lows)
        rsi_high_idx = np.argmax(rsi_values)
        rsi_low_idx = np.argmin(rsi_values)

        # Check for bearish divergence (price higher but RSI lower)
        bearish_div = (price_high_idx > rsi_high_idx) and (
                price_highs[price_high_idx] > price_highs[rsi_high_idx]) and (
                              rsi_values[price_high_idx] < rsi_values[rsi_high_idx])

        # Check for bullish divergence (price lower but RSI higher)
        bullish_div = (price_low_idx > rsi_low_idx) and (price_lows[price_low_idx] < price_lows[rsi_low_idx]) and (
                rsi_values[price_low_idx] > rsi_values[rsi_low_idx])

        return bullish_div, bearish_div


###############################################################################
# ENHANCED BACKTESTING FRAMEWORK
###############################################################################
class EnhancedStrategyBacktester:
    def __init__(self, data_df, preparer, modeler, signal_producer, risk_manager,
                 train_window_size=5000, test_window_size=1000,
                 fixed_cost=0.001, variable_cost=0.0005, slippage=0.0005,
                 walk_forward_steps=4, monte_carlo_sims=100):
        self.data_df = data_df
        self.preparer = preparer
        self.modeler = modeler
        self.signal_producer = signal_producer
        self.risk_manager = risk_manager
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.fixed_cost = fixed_cost  # Fixed cost per trade
        self.variable_cost = variable_cost  # Variable cost as % of trade value
        self.slippage = slippage  # Slippage as % of price
        self.walk_forward_steps = walk_forward_steps
        self.monte_carlo_sims = monte_carlo_sims
        self.results = []
        self.logger = self._setup_logger()

        # Dynamically adjust window sizes based on available data
        self._adjust_window_sizes()

    def _setup_logger(self):
        """Set up logging for the backtester"""
        logger = logging.getLogger("EnhancedBacktester")
        logger.setLevel(logging.INFO)

        # Create handlers
        file_handler = logging.FileHandler(
            f"EnhancedTrainingResults/BacktestLog/backtest_log_{datetime.now():%Y%m%d_%H%M%S}.log")
        file_handler.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)

        return logger

    def _adjust_window_sizes(self):
        """Dynamically adjust window sizes based on available data"""
        df_len = len(self.data_df)

        # Define minimum required data - reduce for small datasets
        min_required = 50  # Minimum dataset size to attempt backtesting

        if df_len < min_required:
            self.logger.warning(f"Insufficient data ({df_len} rows). Need at least {min_required} rows.")
            return False

        # For small datasets, use percentage-based windows
        if df_len < 500:  # Small dataset threshold
            # Adjust window sizes based on available data
            orig_train = self.train_window_size
            orig_test = self.test_window_size

            # Use smaller percentages for tiny datasets to ensure both train and test have data
            self.train_window_size = max(int(df_len * 0.5), 30)  # At least 30 rows for training (50% of data)
            self.test_window_size = max(int(df_len * 0.3), 20)  # At least 20 rows for testing (30% of data)

            self.logger.info(
                f"Adjusted window sizes for small dataset: train_window={orig_train}->{self.train_window_size}, "
                f"test_window={orig_test}->{self.test_window_size}")
        return True

    def walk_forward_backtest(self):
        """Enhanced walk-forward backtest optimized for large datasets"""
        start_idx = 0
        df_len = len(self.data_df)
        iteration = 0

        # Create directory for results
        results_dir = f"EnhancedTrainingResults/Trades"
        os.makedirs(results_dir, exist_ok=True)

        # Store results directly to disk instead of keeping all in memory
        results_path = f"{results_dir}/trades_{datetime.now():%Y%m%d_%H%M%S}.csv"

        # Create empty trades file with headers
        with open(results_path, 'w') as f:
            f.write(
                "iteration,entry_time,exit_time,direction,entry_price,exit_price,quantity,PnL,entry_signal,exit_signal,regime,stop_loss,take_profit\n")

        # Track summary results
        performance_by_iteration = []

        # Use appropriate step size for large datasets
        step_size = self.test_window_size // 2  # 50% overlap between test windows

        # Keep track of overall results to return at the end
        all_results = []

        while start_idx + self.train_window_size + self.test_window_size <= df_len:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration} of walk-forward backtest")
            # Log memory at start of each iteration
            log_memory_usage()

            train_end = start_idx + self.train_window_size
            test_end = min(train_end + self.test_window_size, df_len)  # Make sure we don't exceed data length

            # Get training and testing data
            df_train = self.data_df.iloc[start_idx:train_end].copy()
            df_test = self.data_df.iloc[train_end:test_end].copy()

            # Detect market regime in this period
            regime = self._detect_regime(df_train)
            self.logger.info(f"Detected market regime: {regime}")

            # Prepare data
            X_train, y_train, X_val, y_val, df_val, fwd_returns_val = self.preparer.prepare_data(df_train)

            if len(X_train) == 0:
                self.logger.warning(f"Insufficient training data in iteration {iteration}")
                start_idx += step_size
                continue

            # Check memory after data preparation
            memory_watchdog(threshold_gb=14)

            # Compute class weights with emphasis on extreme classes
            y_train_flat = np.argmax(y_train, axis=1)

            # Get unique classes present in the data
            unique_classes = np.unique(y_train_flat)

            # Calculate class weights only for classes that exist in the data
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_flat)

            # Create weight dictionary mapping class indices to weights
            class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

            # Ensure all possible classes have entries in the dictionary
            for cls in range(5):  # We have 5 classes (0-4)
                if cls not in class_weight_dict:
                    # If a class doesn't exist in training data, assign average weight
                    avg_weight = np.mean(list(class_weight_dict.values())) if class_weight_dict else 1.0
                    class_weight_dict[cls] = avg_weight
                    self.logger.warning(
                        f"Class {cls} not found in training data. Assigned average weight: {avg_weight}")

            # Adjust weights based on regime
            if regime == 'trending':
                # In trending regimes, emphasize strong directional moves
                class_weight_dict[0] *= 1.75  # Strongly bearish
                class_weight_dict[4] *= 1.75  # Strongly bullish
            elif regime == 'ranging':
                # In ranging regimes, emphasize mean reversion
                class_weight_dict[1] *= 1.5  # Moderately bearish
                class_weight_dict[3] *= 1.5  # Moderately bullish
            elif regime == 'volatile':
                # In volatile regimes, all signals matter
                for i in range(5):
                    if i in class_weight_dict:
                        class_weight_dict[i] *= 1.0

            # Use full batch size for large datasets
            batch_size = 256
            epochs = 32

            self.logger.info(f"Using batch_size={batch_size}, epochs={epochs} for training")

            # Train model or ensemble
            if hasattr(self.modeler, 'build_ensemble') and self.modeler.ensemble_size > 1:
                self.logger.info(f"Training ensemble models for iteration {iteration}")
                self.modeler.build_ensemble(X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                                            epochs=epochs, batch_size=batch_size,
                                            class_weight=class_weight_dict)
            else:
                self.logger.info(f"Training single model for iteration {iteration}")
                self.modeler.tune_and_train(iteration, X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                                            epochs=epochs, batch_size=batch_size,
                                            class_weight=class_weight_dict)

            # Force memory cleanup after training
            memory_watchdog(threshold_gb=14, force_cleanup=True)

            # Evaluate model performance
            if len(X_val) > 0:
                self.logger.info(f"Evaluating model for iteration {iteration}")
                self.modeler.evaluate(X_val, y_val)

            # Clean up to save memory
            del X_train, y_train, X_val, y_val, df_val, fwd_returns_val
            collect()

            # Backtest on test period
            self.logger.info(f"Simulating trading for iteration {iteration}")
            test_eq, test_trades = self._simulate_test(df_test, iteration, regime)

            # Record trades directly to disk instead of keeping in memory
            self._save_trades_to_file(test_trades, results_path)

            # Add iteration summary to results
            iter_result = {
                "iteration": iteration,
                "train_start": start_idx,
                "train_end": train_end,
                "test_end": test_end,
                "final_equity": test_eq,
                "regime": regime
            }
            all_results.append(iter_result)

            # Calculate more detailed performance metrics
            iter_performance = self._calculate_performance_metrics(test_trades, test_eq)
            performance_by_iteration.append({
                "iteration": iteration,
                "train_start": start_idx,
                "train_end": train_end,
                "test_end": test_end,
                "final_equity": test_eq,
                "regime": regime,
                "metrics": iter_performance
            })

            # Save iteration results to disk to enable resuming if crashed
            checkpoint_path = f"{results_dir}/checkpoint_iter_{iteration}.json"
            with open(checkpoint_path, 'w') as f:
                import json
                json.dump(iter_result, f)

            # Clean up trades to avoid memory accumulation
            del test_trades, df_train, df_test
            collect()

            # Force memory cleanup after each iteration
            memory_watchdog(threshold_gb=14, force_cleanup=True)

            # Move to next window using step size
            start_idx += step_size

        # Analyze performance across different periods
        self._analyze_period_performance(performance_by_iteration)

        # Save full performance analysis
        perf_summary_path = f"{results_dir}/performance_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(perf_summary_path, 'w') as f:
            import json
            json.dump([{k: v for k, v in p.items() if k != "metrics"} |
                       {"metrics_summary": {metric: val for metric, val in p.get("metrics", {}).items()
                                            if metric in ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown",
                                                          "return"]}}
                       for p in performance_by_iteration], f, default=str)

        # Clean up before returning
        memory_watchdog(threshold_gb=14, force_cleanup=True)

        if not all_results:
            self.logger.warning("No backtest iterations were completed")
            return pd.DataFrame()  # Return empty dataframe if no iterations completed

        return pd.DataFrame(all_results)

    def _save_trades_to_file(self, trades, filepath):
        """Save trades to CSV file to avoid keeping them all in memory"""
        if not trades:
            return

        # Append mode to add trades from multiple iterations
        with open(filepath, 'a') as f:
            for trade in trades:
                # Format trade details safely with default values for missing keys
                line = (
                    f"{trade.get('iteration', 0)},"
                    f"{trade.get('entry_time', '')},"
                    f"{trade.get('exit_time', '')},"
                    f"{trade.get('direction', '')},"
                    f"{trade.get('entry_price', 0)},"
                    f"{trade.get('exit_price', 0)},"
                    f"{trade.get('quantity', 0)},"
                    f"{trade.get('PnL', 0)},"
                    f"{trade.get('entry_signal', '').replace(',', ';')},"  # Replace commas in text fields
                    f"{trade.get('exit_signal', '').replace(',', ';')},"  # to avoid breaking CSV format
                    f"{trade.get('regime', '').replace(',', ';')},"
                    f"{trade.get('stop_loss', 0)},"
                    f"{trade.get('take_profit', 0)}\n"
                )
                f.write(line)

        # Log trade count
        self.logger.info(f"Saved {len(trades)} trades to {filepath}")

        # Optional: Periodically analyze trades file size
        if os.path.exists(filepath):
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if file_size_mb > 100:  # Over 100MB
                self.logger.warning(f"Trade file size is large: {file_size_mb:.2f}MB")

    def _detect_regime(self, df):
        """Detect market regime in the given dataframe"""
        if len(df) < 100:
            return "unknown"

        # Check if we have regime columns from feature engineering
        if 'market_regime' in df.columns and 'volatility_regime' in df.columns:
            # Use the pre-calculated regime features
            market_regime = df['market_regime'].iloc[-20:].mean()
            volatility_regime = df['volatility_regime'].iloc[-20:].mean()

            if volatility_regime > 0.5:
                return "volatile"
            elif abs(market_regime) > 0.5:
                return "trending"
            else:
                return "ranging"

        # Otherwise, calculate regime based on price action
        close = df['close'].values

        # Calculate directional movement
        returns = np.diff(close) / close[:-1]

        # Volatility
        volatility = np.std(returns[-50:]) * np.sqrt(252)  # Annualized

        # Trend - use linear regression slope
        x = np.arange(min(50, len(close)))
        slope, _, r_value, _, _ = linregress(x, close[-min(50, len(close)):])

        trend_strength = abs(r_value)
        normalized_slope = slope / close[-1] * 100  # Normalize by current price

        # Determine regime
        if volatility > 0.8:  # High volatility threshold
            return "volatile"
        elif trend_strength > 0.7 and abs(normalized_slope) > 0.1:
            return "trending"
        else:
            return "ranging"

    def _calculate_performance_metrics(self, trades, final_equity):
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade': 0,
                'return': 0
            }

        # Basic metrics
        win_trades = [t for t in trades if t['PnL'] > 0]
        loss_trades = [t for t in trades if t['PnL'] <= 0]

        total_trades = len(trades)
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0

        total_profit = sum(t['PnL'] for t in win_trades)
        total_loss = abs(sum(t['PnL'] for t in loss_trades)) if loss_trades else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Build equity curve
        initial_capital = self.risk_manager.initial_capital
        equity_curve = [initial_capital]

        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['PnL'])

        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

        # Sharpe ratio
        sharpe = 0
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # Maximum drawdown
        peak = equity_curve[0]
        drawdowns = []

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            drawdowns.append(dd)

        max_drawdown = max(drawdowns)

        # Average trade and return
        avg_trade = sum(t['PnL'] for t in trades) / len(trades) if trades else 0
        total_return = (final_equity - initial_capital) / initial_capital * 100

        # Additional metrics
        avg_win = sum(t['PnL'] for t in win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(t['PnL'] for t in loss_trades) / len(loss_trades) if loss_trades else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Calculate average holding time
        holding_times = []
        for trade in trades:
            entry = pd.to_datetime(trade['entry_time'])
            exit = pd.to_datetime(trade['exit_time'])
            holding_time = (exit - entry).total_seconds() / 3600  # in hours
            holding_times.append(holding_time)

        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'return': total_return,
            'avg_holding_time': avg_holding_time
        }

    def _analyze_period_performance(self, performance_by_iteration):
        """Analyze performance across different time periods and regimes"""
        if not performance_by_iteration:
            return

        # Group by regime
        regime_groups = {}
        for perf in performance_by_iteration:
            regime = perf['regime']
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(perf)

        # Calculate average metrics by regime
        regime_metrics = {}

        for regime, performances in regime_groups.items():
            # Calculate average metrics
            avg_metrics = {}
            for metric in performances[0]['metrics']:
                values = [p['metrics'][metric] for p in performances]
                avg_metrics[metric] = sum(values) / len(values)

            regime_metrics[regime] = avg_metrics

        # Log the analysis
        self.logger.info("\n=== Regime Performance Analysis ===")
        for regime, metrics in regime_metrics.items():
            self.logger.info(f"\nRegime: {regime}")
            for metric, value in metrics.items():
                self.logger.info(f"Avg {metric}: {value}")

        return regime_metrics

    def _run_monte_carlo(self, trades_df, num_simulations=100):
        """Run Monte Carlo simulations to assess strategy robustness"""
        if trades_df.empty:
            return {}

        # Extract PnL from trades
        pnl_values = trades_df['PnL'].values
        initial_capital = self.risk_manager.initial_capital

        # Arrays to store results
        final_equities = np.zeros(num_simulations)
        max_drawdowns = np.zeros(num_simulations)
        sharpe_ratios = np.zeros(num_simulations)

        for i in range(num_simulations):
            # Shuffle the trade sequence
            np.random.shuffle(pnl_values)

            # Build equity curve
            equity = np.zeros(len(pnl_values) + 1)
            equity[0] = initial_capital

            for j in range(len(pnl_values)):
                equity[j + 1] = equity[j] + pnl_values[j]

            # Calculate metrics
            final_equities[i] = equity[-1]

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdowns[i] = drawdown.max()

            # Returns for Sharpe
            returns = np.diff(equity) / equity[:-1]
            sharpe_ratios[i] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Calculate percentiles
        results = {
            'monte_carlo_runs': num_simulations,
            'final_equity_median': np.median(final_equities),
            'final_equity_95pct': np.percentile(final_equities, 95),
            'final_equity_5pct': np.percentile(final_equities, 5),
            'max_drawdown_median': np.median(max_drawdowns),
            'max_drawdown_95pct': np.percentile(max_drawdowns, 95),
            'sharpe_ratio_median': np.median(sharpe_ratios),
            'probability_profit': np.mean(final_equities > initial_capital),
            'probability_10pct_return': np.mean((final_equities - initial_capital) / initial_capital > 0.1),
            'probability_20pct_drawdown': np.mean(max_drawdowns > 0.2)
        }

        return results

    def _simulate_test(self, df_test, iteration, regime="unknown"):
        """Simulate trading on test data"""
        # Prepare test data
        X_test, y_test, df_labeled, _ = self.preparer.prepare_test_data(df_test)
        if len(X_test) == 0:
            self.logger.warning(f"Insufficient test data in iteration {iteration}")
            return self.risk_manager.initial_capital, []

        # Get predictions - handle both ensemble and single model
        # Check if ensemble models exist and are available
        has_ensemble = (hasattr(self.modeler, 'predict_with_ensemble') and
                        hasattr(self.modeler, 'ensemble_models') and
                        self.modeler.ensemble_models)

        if has_ensemble:
            self.logger.info("Using ensemble model for predictions")
            try:
                preds, uncertainties = self.modeler.predict_with_ensemble(X_test)
            except Exception as e:
                self.logger.warning(f"Ensemble prediction failed: {e}, falling back to single model")
                preds = self.modeler.predict_signals(X_test)
                uncertainties = None
        else:
            self.logger.info("Using single model for predictions")
            preds = self.modeler.predict_signals(X_test)
            uncertainties = None

        # Reset risk manager for new test period
        self.risk_manager.current_capital = self.risk_manager.initial_capital
        self.risk_manager.open_positions = []
        self.risk_manager.trade_history = []

        equity_curve = [self.risk_manager.current_capital]
        trades = []
        sequence_length = self.preparer.sequence_length

        # Tracking variables
        position = 0
        trade_entry = None

        # Trade management
        for i in range(len(preds)):
            current_row = i + sequence_length - 1
            if current_row >= len(df_test):
                break

            current_time = df_test.index[current_row]
            current_price = df_test['close'].iloc[current_row]

            # Get signal
            model_probs = preds[i]
            signal_uncertainty = uncertainties[i].max() if uncertainties is not None else 0

            # Adapt confidence threshold based on regime
            confidence_threshold = self.signal_producer.confidence_threshold
            if regime == "volatile":
                confidence_threshold *= 1.2  # Require higher confidence in volatile regimes
            elif regime == "ranging":
                confidence_threshold *= 0.9  # Can be slightly less strict in ranging markets

            # Adjust signal producer params for the current regime
            self.signal_producer.confidence_threshold = confidence_threshold

            # Get signal with potentially modified parameters
            signal = self.signal_producer.get_signal(model_probs, df_test.iloc[:current_row + 1])

            # Log signal
            self.logger.info(f"{current_time} - Signal: {signal.get('signal_type', 'N/A')}, "
                             f"Confidence: {signal.get('confidence', 'N/A')}, "
                             f"Uncertainty: {signal_uncertainty:.4f}")

            # Position management
            if position != 0:
                # Get current ATR for dynamic exit management
                atr_series = self.signal_producer._compute_atr(df_test)
                current_atr = atr_series.iloc[current_row] if current_row < len(atr_series) else 0

                # Check for dynamic exit conditions
                exit_decision = self.risk_manager.dynamic_exit_strategy(trade_entry, current_price, current_atr)

                # Handle trailing stops
                if exit_decision.get('update_stop', False):
                    trade_entry['stop_loss'] = exit_decision['new_stop']
                    self.logger.info(f"{current_time} - Updated stop to {trade_entry['stop_loss']:.2f}")

                # Handle partial exits
                if exit_decision.get('partial_exit', False):
                    exit_ratio = exit_decision.get('exit_ratio', 0.5)
                    exit_reason = exit_decision.get('reason', 'PartialExit')

                    close_quantity = trade_entry['quantity'] * exit_ratio
                    proportion = close_quantity / trade_entry['quantity']
                    entry_cost_alloc = proportion * trade_entry['total_entry_cost']

                    # Calculate exit costs with slippage
                    slippage_amount = current_price * self.slippage
                    exit_price = current_price - slippage_amount if position > 0 else current_price + slippage_amount
                    exit_cost = self.fixed_cost + (exit_price * close_quantity * self.variable_cost)

                    # Calculate PnL
                    pnl_partial = close_quantity * (exit_price - trade_entry[
                        'entry_price']) - entry_cost_alloc - exit_cost if position > 0 else \
                        close_quantity * (trade_entry['entry_price'] - exit_price) - entry_cost_alloc - exit_cost

                    # Update position
                    self.risk_manager.current_capital += pnl_partial
                    trade_entry['quantity'] -= close_quantity
                    trade_entry['total_entry_cost'] -= entry_cost_alloc

                    if exit_reason == "First target":
                        trade_entry['partial_exit_1'] = True
                    elif exit_reason == "Second target":
                        trade_entry['partial_exit_2'] = True

                    # Record partial trade
                    partial_trade_record = trade_entry.copy()
                    partial_trade_record.update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'exit_signal': exit_reason,
                        'exit_confidence': None,
                        'PnL': pnl_partial,
                        'quantity': close_quantity,
                        'regime': regime
                    })
                    trades.append(partial_trade_record)

                    self.logger.info(
                        f"{current_time} - Partial close {exit_ratio * 100:.0f}% at {exit_price:.2f}, PnL: {pnl_partial:.2f}")

                    # Record trade in risk manager history
                    self.risk_manager.trade_history.append({
                        'entry_time': trade_entry['entry_time'],
                        'exit_time': current_time,
                        'direction': trade_entry['direction'],
                        'pnl': pnl_partial,
                        'exit_reason': exit_reason
                    })

                # Check for full exit conditions
                hit_stop = (position > 0 and current_price <= trade_entry['stop_loss']) or \
                           (position < 0 and current_price >= trade_entry['stop_loss'])

                signal_reversal = (position > 0 and "Sell" in signal['signal_type']) or \
                                  (position < 0 and "Buy" in signal['signal_type'])

                if hit_stop or signal_reversal:
                    # Process full exit
                    exit_time = current_time
                    slippage_amount = current_price * self.slippage
                    exit_price = current_price - slippage_amount if position > 0 and signal_reversal else \
                        current_price + slippage_amount if position < 0 and signal_reversal else \
                            trade_entry['stop_loss']  # Use exact stop price for stop loss hits

                    exit_signal = signal['signal_type'] if signal_reversal else "StopLoss"
                    exit_confidence = signal.get('confidence', None) if signal_reversal else None

                    # Calculate costs and PnL
                    quantity = trade_entry['quantity']
                    cost_entry = trade_entry['total_entry_cost']
                    cost_exit = self.fixed_cost + (exit_price * quantity * self.variable_cost)

                    PnL = quantity * (
                            exit_price - trade_entry['entry_price']) - cost_entry - cost_exit if position > 0 else \
                        quantity * (trade_entry['entry_price'] - exit_price) - cost_entry - cost_exit

                    # Update capital
                    self.risk_manager.current_capital += PnL

                    # Record trade
                    trade_record = trade_entry.copy()
                    trade_record.update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'exit_signal': exit_signal,
                        'exit_confidence': exit_confidence,
                        'PnL': PnL,
                        'quantity': quantity,
                        'regime': regime
                    })
                    trades.append(trade_record)

                    self.logger.info(f"{exit_time} - Closing {trade_entry['direction']} at {exit_price:.2f}, "
                                     f"PnL: {PnL:.2f}, Reason: {exit_signal}")

                    # Record trade in risk manager history
                    self.risk_manager.trade_history.append({
                        'entry_time': trade_entry['entry_time'],
                        'exit_time': exit_time,
                        'direction': trade_entry['direction'],
                        'pnl': PnL,
                        'exit_reason': exit_signal
                    })

                    # Reset position
                    position = 0
                    trade_entry = None

            # Check for entry conditions
            if position == 0 and ("Buy" in signal['signal_type'] or "Sell" in signal['signal_type']):
                direction = 'long' if "Buy" in signal['signal_type'] else 'short'
                confidence = signal['confidence']

                # Check if confidence high enough given uncertainty
                if uncertainties is not None:
                    uncertainty = signal_uncertainty
                    if uncertainty > 0.3:  # High uncertainty threshold
                        confidence *= (1 - uncertainty)

                    if confidence < confidence_threshold:
                        self.logger.info(f"{current_time} - Signal rejected due to high uncertainty")
                        continue

                # Check correlation risk for the new position
                can_add_position, adjusted_risk = self.risk_manager.check_correlation_risk({
                    'direction': direction,
                    'risk_amount': self.risk_manager.current_capital * self.risk_manager.max_risk_per_trade
                })

                if not can_add_position:
                    self.logger.info(f"{current_time} - Signal rejected due to correlation risk")
                    continue

                # Determine stop loss
                if 'stop_loss' in signal:
                    stop_loss = signal['stop_loss']
                else:
                    atr = self.signal_producer._compute_atr(df_test).iloc[current_row] if current_row < len(
                        df_test) else 0
                    distance = self.signal_producer.atr_multiplier_sl * atr
                    stop_loss = current_price - distance if direction == 'long' else current_price + distance

                # Calculate position size with risk manager
                volatility_regime = signal.get('volatility_regime', 0)
                quantity = self.risk_manager.calculate_position_size(
                    signal, current_price, stop_loss, volatility_regime)

                if quantity <= 0:
                    self.logger.info(f"{current_time} - Zero position size calculated")
                    continue

                # Add slippage to entry
                slippage_amount = current_price * self.slippage
                entry_price = current_price + slippage_amount if direction == 'long' else current_price - slippage_amount

                # Calculate entry costs
                total_entry_cost = self.fixed_cost + (entry_price * quantity * self.variable_cost)

                # Set take profit levels
                take_profit_dist = abs(entry_price - stop_loss) * self.risk_manager.reward_risk_ratio
                take_profit = entry_price + take_profit_dist if direction == 'long' else entry_price - take_profit_dist

                # Create partial take profit levels
                partial_close_price1 = entry_price + (
                        take_profit_dist * 0.25) if direction == 'long' else entry_price - (take_profit_dist * 0.25)
                partial_close_price2 = entry_price + (
                        take_profit_dist * 0.5) if direction == 'long' else entry_price - (take_profit_dist * 0.5)

                # Open position
                position = 1 if direction == 'long' else -1
                entry_time = current_time

                # Record trade entry
                trade_entry = {
                    'iteration': iteration,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'direction': direction,
                    'entry_signal': signal['signal_type'],
                    'entry_confidence': signal['confidence'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'partial_close_price': partial_close_price1,
                    'partial_close_price2': partial_close_price2,
                    'quantity': quantity,
                    'initial_distance': abs(entry_price - stop_loss),
                    'partial_exit_1': False,
                    'partial_exit_2': False,
                    'total_entry_cost': total_entry_cost,
                    'regime': regime
                }

                # Log the entry
                self.logger.info(f"{entry_time} - Opening {direction} at {entry_price:.2f}, "
                                 f"qty: {quantity:.4f}, signal: {signal['signal_type']}, "
                                 f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

                # Add to risk manager's open positions
                self.risk_manager.open_positions.append({
                    'direction': direction,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'quantity': quantity,
                    'risk_amount': quantity * abs(entry_price - stop_loss)
                })

            # Record equity
            equity_curve.append(self.risk_manager.current_capital)

        # Close any open position at the end of test
        if position != 0 and trade_entry is not None:
            exit_time = df_test.index[-1]
            exit_price = df_test['close'].iloc[-1]
            exit_signal = "EndOfTest"

            # Calculate costs and PnL
            quantity = trade_entry['quantity']
            cost_entry = trade_entry['total_entry_cost']
            cost_exit = self.fixed_cost + (exit_price * quantity * self.variable_cost)

            PnL = quantity * (exit_price - trade_entry['entry_price']) - cost_entry - cost_exit if position > 0 else \
                quantity * (trade_entry['entry_price'] - exit_price) - cost_entry - cost_exit

            # Update capital
            self.risk_manager.current_capital += PnL

            # Record trade
            trade_record = trade_entry.copy()
            trade_record.update({
                'exit_time': exit_time,
                'exit_price': exit_price,
                'exit_signal': exit_signal,
                'exit_confidence': None,
                'PnL': PnL,
                'quantity': quantity,
                'regime': regime
            })
            trades.append(trade_record)

            self.logger.info(f"{exit_time} - Closing {trade_entry['direction']} at {exit_price:.2f}, PnL: {PnL:.2f}")

            # Record in risk manager
            self.risk_manager.trade_history.append({
                'entry_time': trade_entry['entry_time'],
                'exit_time': exit_time,
                'direction': trade_entry['direction'],
                'pnl': PnL,
                'exit_reason': exit_signal
            })

        # Update risk manager performance stats
        self.risk_manager.update_performance_stats()

        return self.risk_manager.current_capital, trades