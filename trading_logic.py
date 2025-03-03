import logging
import os
from datetime import datetime
from gc import collect

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.utils import compute_class_weight

from memory_utils import (
    memory_watchdog,
    log_memory_usage
)


###############################################################################
# ADVANCED RISK MANAGEMENT
###############################################################################
class AdvancedRiskManager:
    def __init__(self, initial_capital=10000.0, max_risk_per_trade=0.025,
                 max_correlated_exposure=0.08, volatility_scaling=True,
                 target_annual_vol=0.25):
        """
        Class providing advanced risk management features for short-term trading.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Risk parameters
        self.max_risk_per_trade = max_risk_per_trade
        self.max_correlated_exposure = max_correlated_exposure
        self.volatility_scaling = volatility_scaling
        self.target_annual_vol = target_annual_vol
        self.reward_risk_ratio = 2.0
        self.partial_profit_points = [0.25, 0.4, 0.6]
        self.partial_profit_sizes = [0.3, 0.3, 0.4]
        self.max_trades_per_day = 5
        self.max_drawdown_percent = 0.15
        self.max_consecutive_losses = 4
        self.strong_signal_threshold = 0.7
        self.confidence_threshold = 0.5

        # Positions and trade history
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

        self.logger = logging.getLogger("RiskManager")

    def calculate_position_size(self, signal, entry_price, stop_loss, volatility_regime=0, funding_rate=0):
        """
        Calculate optimal position size using volatility-adjusted Kelly criterion,
        with adaptive modifications based on performance, volatility, and funding.
        """
        if entry_price == stop_loss:
            return 0

        # Capital-based risk scaling
        capital_factor = self._get_capital_factor()
        base_risk_pct = self.max_risk_per_trade * capital_factor

        # Adjust risk based on market regime
        market_regime = signal.get('regime', 0)
        base_risk_pct = self._apply_regime_factor(base_risk_pct, market_regime)

        # Performance-based risk adjustment
        base_risk_pct = self._apply_performance_factor(base_risk_pct)

        # Volatility-based risk adjustment
        if self.volatility_scaling:
            base_risk_pct = self._apply_volatility_factor(base_risk_pct, volatility_regime, entry_price, stop_loss)

        # Funding rate adjustment
        base_risk_pct = self._apply_funding_factor(base_risk_pct, funding_rate, signal)

        # Signal confidence weighting
        confidence = signal.get('confidence', 0.5)
        base_risk_pct = self._apply_confidence_factor(base_risk_pct, confidence)

        # Enforce maximum risk limit with slight allowance for strong signals
        risk_pct = min(base_risk_pct, self.max_risk_per_trade * base_risk_pct / self.max_risk_per_trade)

        # Combine with Kelly fraction for a final safety-limited risk
        final_risk_pct = self._apply_kelly_criterion(signal, confidence, risk_pct)

        # Enforce a minimum viable position size
        min_viable_pct = 0.005
        if final_risk_pct < min_viable_pct:
            return 0

        risk_amount = self.current_capital * final_risk_pct * 1.02
        risk_per_unit = abs(entry_price - stop_loss)

        position_size = 0
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit

        self.logger.info(
            f"Position sizing => capital_factor={capital_factor:.2f}, "
            f"regime={market_regime:.2f}, final_risk_pct={final_risk_pct:.4f}, "
            f"size={position_size:.6f}"
        )
        return position_size

    def check_correlation_risk(self, new_signal):
        """
        Check if adding this position would exceed maximum risk for correlated assets.
        Since we typically trade a single asset (BTC), this is often a simple check.
        """
        current_exposure = sum(pos['risk_amount'] for pos in self.open_positions)
        current_exposure_pct = current_exposure / self.current_capital

        if current_exposure_pct + self.max_risk_per_trade > self.max_correlated_exposure:
            return False, self.max_correlated_exposure - current_exposure_pct

        return True, self.max_risk_per_trade

    def dynamic_exit_strategy(self, position, current_price, current_atr, funding_rate=0):
        """
        Implement market-adaptive exit strategy with dynamic thresholds.
        Applies partial exits, trailing stops, and final exit logic.
        """
        direction = position['direction']
        entry_price = position['entry_price']

        # Calculate current profit or loss (as %)
        pnl_pct = self._calculate_pnl_percent(direction, current_price, entry_price)
        market_regime = position.get('market_regime', 'unknown')
        volatility_regime = position.get('volatility_regime', 0)

        # Determine general targets/ratios
        tp_levels, exit_ratios = self._compute_targets_and_exit_ratios(
            entry_price, market_regime, volatility_regime, current_atr, direction
        )

        # Check partial exit conditions
        partial_exit = self._check_partial_exits(position, current_price, tp_levels, exit_ratios, direction)
        if partial_exit:
            return partial_exit

        # Check final target conditions
        final_exit = self._check_final_target_and_funding(
            position, current_price, current_atr, volatility_regime,
            funding_rate, entry_price, direction, tp_levels
        )
        if final_exit:
            return final_exit

        # If no partial or final exit triggered, check for trailing stop updates
        trailing_stop_update = self._check_trailing_stop(
            position, current_price, current_atr, market_regime,
            volatility_regime, entry_price, pnl_pct, direction
        )
        return trailing_stop_update

    def update_performance_stats(self):
        """
        Update performance statistics based on trade history.
        """
        if not self.trade_history:
            return

        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] < 0]

        self.performance_stats['win_rate'] = (len(wins) / len(self.trade_history)) if self.trade_history else 0
        self.performance_stats['avg_win'] = (sum(t['pnl'] for t in wins) / len(wins)) if wins else 0
        self.performance_stats['avg_loss'] = (sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
        self.performance_stats['largest_win'] = max((t['pnl'] for t in wins), default=0)
        self.performance_stats['largest_loss'] = min((t['pnl'] for t in losses), default=0)

        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        self.performance_stats['profit_factor'] = total_profit / total_loss if total_loss else float('inf')

        # Calculate equity curve, drawdown, returns
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

    ###########################################################################
    # PRIVATE / HELPER METHODS
    ###########################################################################
    def _get_capital_factor(self):
        """
        Adjust capital-based risk factor depending on drawdowns or large gains.
        """
        if self.current_capital > self.initial_capital * 3:
            return 0.85
        elif self.current_capital > self.initial_capital * 2:
            return 0.9
        elif self.current_capital < self.initial_capital * 0.7:
            return 0.7
        return 1.0

    def _apply_regime_factor(self, base_risk_pct, market_regime):
        """
        Adjust risk based on market regime.
        """
        if abs(market_regime) > 0.8:  # Strong trend
            regime_factor = 1.1
        elif abs(market_regime) < 0.2:  # Weak or no trend
            regime_factor = 0.9
        else:
            regime_factor = 1.0
        return base_risk_pct * regime_factor

    def _apply_performance_factor(self, base_risk_pct):
        """
        Adapt risk based on recent performance and streaks.
        """
        if len(self.trade_history) < 10:
            return base_risk_pct

        recent_trades = self.trade_history[-10:]
        win_count = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        win_rate = win_count / len(recent_trades)
        consecutive_count = 1

        for i in range(len(recent_trades) - 2, -1, -1):
            curr_profitable = recent_trades[i + 1].get('pnl', 0) > 0
            prev_profitable = recent_trades[i].get('pnl', 0) > 0
            if curr_profitable == prev_profitable:
                consecutive_count += 1
            else:
                break

        last_trade_profitable = recent_trades[-1].get('pnl', 0) > 0
        performance_factor = 1.0

        if last_trade_profitable:
            if consecutive_count >= 4:  # winning streak
                if win_rate > 0.7:
                    performance_factor = min(1.3, 1.0 + (consecutive_count - 3) * 0.1)
                else:
                    performance_factor = min(1.15, 1.0 + (consecutive_count - 3) * 0.05)
        else:
            if consecutive_count >= 3:  # losing streak
                performance_factor = max(0.6, 1.0 - (consecutive_count - 2) * 0.1)
            elif win_rate < 0.4:
                performance_factor = 0.8

        return base_risk_pct * performance_factor

    def _apply_volatility_factor(self, base_risk_pct, volatility_regime, entry_price, stop_loss):
        """
        Adjust risk for high/low volatility regimes or ATR-based estimates.
        """
        if volatility_regime != 0:
            # Scale between 0.75 and 1.25 based on volatility_regime
            vol_factor = 1.0 - (volatility_regime * 0.25)
        else:
            # Fallback: estimate via ATR
            atr_pct = abs(entry_price - stop_loss) / entry_price
            if atr_pct > 0.03:
                vol_factor = 0.7
            elif atr_pct > 0.02:
                vol_factor = 0.8
            elif atr_pct < 0.01:
                vol_factor = 1.3
            elif atr_pct < 0.005:
                vol_factor = 1.5
            else:
                vol_factor = 1.0

        return base_risk_pct * vol_factor

    def _apply_funding_factor(self, base_risk_pct, funding_rate, signal):
        """
        Adjust risk based on funding rates. More extreme rates => bigger adjustments.
        """
        funding_factor = 1.0
        direction = 'long' if signal.get('signal_type', '').startswith('Buy') else 'short'
        magnitude = abs(funding_rate) / 0.0001  # normalized around 0.01% typical

        if direction == 'long':
            if funding_rate < -0.0001:  # negative funding favors longs
                funding_factor = min(1.5, 1.0 + (magnitude * 0.1))
            elif funding_rate > 0.0003:
                funding_factor = max(0.7, 1.0 - (magnitude * 0.05))
        else:  # short
            if funding_rate > 0.0001:  # positive funding favors shorts
                funding_factor = min(1.5, 1.0 + (magnitude * 0.1))
            elif funding_rate < -0.0003:
                funding_factor = max(0.7, 1.0 - (magnitude * 0.05))

        return base_risk_pct * funding_factor

    def _apply_confidence_factor(self, base_risk_pct, confidence):
        """
        Weight position size by model confidence. Strong signals can get slight increase.
        """
        confidence_factor = 1.0
        if confidence > self.strong_signal_threshold:
            # Cap at 30% increase
            confidence_factor = min(1.3, 1.0 + ((confidence - self.strong_signal_threshold) * 2))
        elif confidence < self.confidence_threshold * 1.1:
            # Reduce size for marginal signals
            confidence_ratio = confidence / self.confidence_threshold
            confidence_factor = max(0.8, confidence_ratio)

        return base_risk_pct * confidence_factor

    def _apply_kelly_criterion(self, signal, confidence, risk_pct):
        """
        Combine simplified Kelly fraction with a safety factor (0.5).
        """
        b = signal.get('reward_risk_ratio', 2.0)
        p = confidence
        q = 1 - p
        kelly_fraction = max(0, (b * p - q) / b) if b > 0 else 0

        kelly_factor = 0.5
        kelly_adjusted_risk = kelly_fraction * self.max_risk_per_trade * kelly_factor
        return min(risk_pct, kelly_adjusted_risk)

    def _calculate_pnl_percent(self, direction, current_price, entry_price):
        """
        Helper to calculate the current PnL in percent terms.
        """
        if direction == 'long':
            return (current_price - entry_price) / entry_price
        return (entry_price - current_price) / entry_price

    def _compute_targets_and_exit_ratios(self, entry_price, market_regime, volatility_regime,
                                         current_atr, direction):
        """
        Compute the various profit targets and partial exit ratios,
        factoring in volatility/regime.
        """
        # Base TP ~1% & scale it
        base_tp_pct = 0.01
        # Use ATR as a volatility measure
        atr_pct = current_atr / entry_price if entry_price else 0
        vol_factor = max(0.5, min(2.0, atr_pct / 0.01))
        if volatility_regime != 0:
            vol_factor *= (1.0 + (volatility_regime * 0.3))

        regime_factor = 1.0
        if market_regime == 'trending':
            regime_factor = 1.3
        elif market_regime == 'ranging':
            regime_factor = 0.8
        elif market_regime == 'volatile':
            regime_factor = 1.5

        tp1_pct = base_tp_pct * vol_factor * regime_factor
        tp2_pct = tp1_pct * 2.0
        tp3_pct = tp1_pct * 3.0
        tp4_pct = tp1_pct * 4.0

        if direction == 'long':
            targets = [
                entry_price * (1 + tp1_pct),
                entry_price * (1 + tp2_pct),
                entry_price * (1 + tp3_pct),
                entry_price * (1 + tp4_pct)
            ]
        else:  # short
            targets = [
                entry_price * (1 - tp1_pct),
                entry_price * (1 - tp2_pct),
                entry_price * (1 - tp3_pct),
                entry_price * (1 - tp4_pct)
            ]

        # Base partial exit ratios; tweak by regime
        base_exit1, base_exit2, base_exit3 = 0.25, 0.33, 0.5
        if market_regime == 'trending':
            exit_ratios = [base_exit1 * 0.8, base_exit2 * 0.9, base_exit3 * 0.9]
        elif market_regime == 'ranging':
            exit_ratios = [base_exit1 * 1.2, base_exit2 * 1.1, base_exit3 * 1.0]
        else:
            exit_ratios = [base_exit1, base_exit2, base_exit3]

        return targets, exit_ratios

    def _check_partial_exits(self, position, current_price, targets, exit_ratios, direction):
        """
        Check each partial exit target for potential partial closure and updated stop.
        """
        initial_stop = position['stop_loss']
        # Process partial exit triggers in order
        # 1st target
        if not position.get('partial_exit_1', False):
            if (direction == 'long' and current_price >= targets[0]) \
               or (direction == 'short' and current_price <= targets[0]):
                new_stop = self._compute_partial_exit_stop(direction, initial_stop, position['entry_price'], 1)
                return {
                    "partial_exit": True,
                    "exit_ratio": exit_ratios[0],
                    "reason": "FirstTarget",
                    "update_stop": True,
                    "new_stop": new_stop
                }
        # 2nd target
        if not position.get('partial_exit_2', False):
            if (direction == 'long' and current_price >= targets[1]) \
               or (direction == 'short' and current_price <= targets[1]):
                new_stop = self._compute_partial_exit_stop(direction, initial_stop, position['entry_price'], 2)
                return {
                    "partial_exit": True,
                    "exit_ratio": exit_ratios[1],
                    "reason": "SecondTarget",
                    "update_stop": True,
                    "new_stop": new_stop
                }
        # 3rd target
        if not position.get('partial_exit_3', False):
            if (direction == 'long' and current_price >= targets[2]) \
               or (direction == 'short' and current_price <= targets[2]):
                new_stop = self._compute_partial_exit_lock(direction, initial_stop, position['entry_price'], targets[2],
                                                           position.get('market_regime', 'unknown'))
                return {
                    "partial_exit": True,
                    "exit_ratio": exit_ratios[2],
                    "reason": "ThirdTarget",
                    "update_stop": True,
                    "new_stop": new_stop
                }

        return None

    def _compute_partial_exit_stop(self, direction, initial_stop, entry_price, target_id):
        """
        Compute updated stop after 1st or 2nd partial exit.
        """
        # Slightly tighter stops after partial exit
        # If 'long', raise stop; if 'short', lower stop.
        # Example approach: for the 2nd partial exit in a trending regime, move stop to near breakeven, etc.
        if target_id == 1:
            # Move stop a bit beyond original
            factor = 0.997 if direction == 'long' else 1.003
            adjusted_stop = entry_price * factor
            return max(initial_stop, adjusted_stop) if direction == 'long' else min(initial_stop, adjusted_stop)

        # If target_id == 2
        return entry_price if direction == 'long' else entry_price

    def _compute_partial_exit_lock(self, direction, initial_stop, entry_price, current_price, market_regime):
        """
        Compute updated stop for third partial exit using a profit lock approach.
        """
        if market_regime == 'trending':
            profit_lock = 0.3
        else:
            profit_lock = 0.5

        profit_amount = (current_price - entry_price) if direction == 'long' else (entry_price - current_price)
        new_stop = entry_price + (profit_amount * profit_lock) if direction == 'long' else entry_price - (profit_amount * profit_lock)

        if direction == 'long':
            return max(initial_stop, new_stop)
        return min(initial_stop, new_stop)

    def _check_final_target_and_funding(self, position, current_price, current_atr, volatility_regime,
                                        funding_rate, entry_price, direction, targets):
        """
        Check if final big target is reached or if funding-based exit triggers a full exit.
        """
        final_target = targets[3]
        if (direction == 'long' and current_price >= final_target) \
           or (direction == 'short' and current_price <= final_target):

            # If funding is strongly against the position, exit fully
            # Otherwise tighten trailing stop
            threshold = 0.0001 * (1 + volatility_regime * 0.3)  # more flexible
            if (direction == 'long' and funding_rate > threshold) or (direction == 'short' and funding_rate < -threshold):
                return {
                    "partial_exit": True,
                    "exit_ratio": 1.0,
                    "reason": "FundingBasedExit"
                }
            else:
                distance_from_entry = abs((current_price / entry_price) - 1)
                # Tighter trailing if we've moved far
                trail_pct = max(0.005, min(0.02, current_atr / current_price * (1.0 - min(0.5, distance_from_entry))))
                new_stop = current_price * (1 - trail_pct) if direction == 'long' else current_price * (1 + trail_pct)

                if direction == 'long' and new_stop > position['stop_loss']:
                    return {"update_stop": True, "new_stop": new_stop}
                elif direction == 'short' and new_stop < position['stop_loss']:
                    return {"update_stop": True, "new_stop": new_stop}

        return None

    def _check_trailing_stop(self, position, current_price, current_atr, market_regime,
                             volatility_regime, entry_price, pnl_pct, direction):
        """
        Dynamically update trailing stop if profit has reached certain threshold.
        """
        min_profit_to_trail = 0.005
        if market_regime == 'trending':
            min_profit_to_trail *= 1.2
        elif market_regime == 'volatile':
            min_profit_to_trail *= 0.8

        if pnl_pct > min_profit_to_trail:
            base_atr_multiple = 1.5
            if volatility_regime > 0:
                atr_multiple = base_atr_multiple * 1.2
            elif volatility_regime < 0:
                atr_multiple = base_atr_multiple * 0.8
            else:
                atr_multiple = base_atr_multiple

            # Tighter trailing as profit grows
            # profit_factor ~ how many times first target is gained
            # Just a simplified approach to gradually reduce the multiple
            # so that the more it runs, the tighter we get.
            # We'll assume first target ~1% from entry, so let's compare:
            profit_factor = pnl_pct / 0.01 if 0.01 != 0 else 1.0
            trail_factor = max(1.0, min(3.0, 1.0 + (profit_factor * 2.0)))
            adaptive_multiple = atr_multiple / trail_factor

            new_stop = current_price - (adaptive_multiple * current_atr) if direction == 'long' \
                else current_price + (adaptive_multiple * current_atr)

            if direction == 'long' and new_stop > position['stop_loss']:
                return {"update_stop": True, "new_stop": new_stop}
            elif direction == 'short' and new_stop < position['stop_loss']:
                return {"update_stop": True, "new_stop": new_stop}

        return {"update_stop": False}


###############################################################################
# REFACTORED SIGNAL GENERATION
###############################################################################
class SignalGenerator:
    """Base class for signal generation with common methods."""

    def __init__(self):
        self.logger = logging.getLogger("SignalGenerator")

    def compute_atr(self, df, period=14):
        """Compute Average True Range indicator."""
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low'] - df['close'].shift(1)).abs()
        }).max(axis=1)
        return tr.rolling(window=period).mean()


class TechnicalSignalGenerator(SignalGenerator):
    """Generates signals based on technical indicators."""

    def __init__(self):
        super().__init__()

    def detect_divergence(self, df):
        """
        Detect price-indicator divergence for additional signals.
        Returns tuple: (bullish_divergence, bearish_divergence).
        """
        if 'h4_RSI_14' not in df.columns or len(df) < 20:
            return False, False

        price_highs = df['high'].iloc[-20:].values
        price_lows = df['low'].iloc[-20:].values
        rsi_values = df['h4_RSI_14'].iloc[-20:].values

        price_high_idx = np.argmax(price_highs)
        price_low_idx = np.argmin(price_lows)
        rsi_high_idx = np.argmax(rsi_values)
        rsi_low_idx = np.argmin(rsi_values)

        # Bearish divergence
        bearish_div = (price_high_idx > rsi_high_idx) and \
                      (price_highs[price_high_idx] > price_highs[rsi_high_idx]) and \
                      (rsi_values[price_high_idx] < rsi_values[rsi_high_idx])

        # Bullish divergence
        bullish_div = (price_low_idx > rsi_low_idx) and \
                      (price_lows[price_low_idx] < price_lows[rsi_low_idx]) and \
                      (rsi_values[price_low_idx] > rsi_values[rsi_low_idx])

        return bullish_div, bearish_div

    def detect_short_term_patterns(self, df):
        """
        Detect short-term candlestick patterns: engulfing, doji, hammer, shooting star.
        Returns dict of pattern flags.
        """
        if len(df) < 5:
            return {
                'bullish_engulfing': False,
                'bearish_engulfing': False,
                'doji': False,
                'hammer': False,
                'shooting_star': False
            }

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Engulfing
        bullish_engulfing = (
            curr['close'] > curr['open'] and
            prev['close'] < prev['open'] and
            curr['close'] > prev['open'] and
            curr['open'] < prev['close']
        )
        bearish_engulfing = (
            curr['close'] < curr['open'] and
            prev['close'] > prev['open'] and
            curr['close'] < prev['open'] and
            curr['open'] > prev['close']
        )

        # Doji
        body_size = abs(curr['close'] - curr['open'])
        candle_range = curr['high'] - curr['low']
        doji = body_size <= (candle_range * 0.1)

        # Hammer
        lower_wick = min(curr['open'], curr['close']) - curr['low']
        upper_wick = curr['high'] - max(curr['open'], curr['close'])
        hammer = (
            body_size <= (candle_range * 0.3) and
            lower_wick >= (body_size * 2) and
            upper_wick <= (body_size * 0.5)
        )

        # Shooting star
        shooting_star = (
            body_size <= (candle_range * 0.3) and
            upper_wick >= (body_size * 2) and
            lower_wick <= (body_size * 0.5)
        )

        return {
            'bullish_engulfing': bullish_engulfing,
            'bearish_engulfing': bearish_engulfing,
            'doji': doji,
            'hammer': hammer,
            'shooting_star': shooting_star
        }


class MarketContextAnalyzer(SignalGenerator):
    """Analyzes market context (regimes, funding, etc.) for signal enhancement."""

    def __init__(self):
        super().__init__()

    def analyze_regime(self, df):
        """
        Determine a numeric market regime score from the DataFrame's known features.
        """
        regime_score = 0

        if 'market_regime' in df.columns:
            regime_score += df['market_regime'].iloc[-1] * 2

        if 'trend_strength' in df.columns:
            regime_score += df['trend_strength'].iloc[-1]

        if 'h4_SMA_20' in df.columns and 'h4_SMA_50' in df.columns:
            sma_20 = df['h4_SMA_20'].iloc[-1]
            sma_50 = df['h4_SMA_50'].iloc[-1]
            regime_score += 0.5 if sma_20 > sma_50 else -0.5

        if 'h4_RSI_14' in df.columns:
            rsi = df['h4_RSI_14'].iloc[-1]
            if rsi > 70:
                regime_score -= 0.5
            elif rsi < 30:
                regime_score += 0.5

        if 'rsi_5' in df.columns:
            rsi_5 = df['rsi_5'].iloc[-1]
            if rsi_5 > 80:
                regime_score -= 0.3
            elif rsi_5 < 20:
                regime_score += 0.3

        return regime_score

    def analyze_funding(self, df, funding_df):
        """
        Analyze funding rate context. Returns a single numeric signal
        (positive => bullish, negative => bearish).
        """
        funding_signal = 0
        if 'funding_rate' in df.columns:
            current_rate = df['funding_rate'].iloc[-1]
            funding_signal = self._calc_funding_signal(current_rate)
        elif funding_df is not None and not funding_df.empty and 'fundingRate' in funding_df.columns:
            current_rate = funding_df['fundingRate'].iloc[-1]
            funding_signal = self._calc_funding_signal(current_rate)
        return funding_signal

    def analyze_volatility(self, df):
        """
        Analyze volatility context for signal adjustment. Returns numeric score [-1..1].
        """
        volatility_score = 0
        if 'volatility_regime' in df.columns:
            volatility_score = df['volatility_regime'].iloc[-1]
        elif 'hist_vol_20' in df.columns:
            hist_vol = df['hist_vol_20'].dropna()
            current_vol = hist_vol.iloc[-1]
            pct_rank = sum(1 for x in hist_vol if x <= current_vol) / len(hist_vol)
            volatility_score = (pct_rank * 2) - 1
        elif 'hist_vol_5' in df.columns:
            hist_vol_5 = df['hist_vol_5'].dropna()
            current_vol_5 = hist_vol_5.iloc[-1]
            if len(hist_vol_5) >= 10:
                recent_vol = hist_vol_5.iloc[-10:].mean()
                if current_vol_5 > recent_vol * 1.5:
                    volatility_score = 0.8
                elif current_vol_5 < recent_vol * 0.7:
                    volatility_score = -0.5
        elif 'd1_ATR_14' in df.columns:
            atr = df['d1_ATR_14'].iloc[-1]
            price = df['close'].iloc[-1]
            atr_pct = atr / price
            if atr_pct > 0.03:
                volatility_score = 1
            elif atr_pct > 0.02:
                volatility_score = 0.5
            elif atr_pct < 0.01:
                volatility_score = -0.5

        return volatility_score

    def analyze_short_term_momentum(self, df):
        """
        Analyze short-term momentum for faster trading signals.
        """
        momentum_score = 0
        if 'momentum_5' in df.columns:
            momentum_5 = df['momentum_5'].iloc[-1]
            if momentum_5 > 0.02:
                momentum_score += 1
            elif momentum_5 < -0.02:
                momentum_score -= 1
            elif momentum_5 > 0.01:
                momentum_score += 0.5
            elif momentum_5 < -0.01:
                momentum_score -= 0.5

        if 'rsi_5' in df.columns:
            rsi_5 = df['rsi_5'].iloc[-1]
            if rsi_5 > 75:
                momentum_score -= 0.7
            elif rsi_5 < 25:
                momentum_score += 0.7
            elif rsi_5 > 65:
                momentum_score -= 0.3
            elif rsi_5 < 35:
                momentum_score += 0.3

        if 'volume_spike' in df.columns and df['volume_spike'].iloc[-1] > 0:
            # Volume spike in direction of momentum
            if 'momentum_5' in df.columns:
                momentum_5 = df['momentum_5'].iloc[-1]
                momentum_score += 0.5 * np.sign(momentum_5)

        return momentum_score

    def _calc_funding_signal(self, current_rate):
        """
        Helper to convert a funding rate into a bullish/bearish numeric signal.
        """
        if current_rate > 0.001:
            return -1
        elif current_rate < -0.001:
            return 1
        elif current_rate > 0.0005:
            return -0.5
        elif current_rate < -0.0005:
            return 0.5
        return 0


class EnhancedSignalProducer:
    """
    Enhanced signal generator with improved confidence filtering
    and better market context integration.
    """

    def __init__(self, confidence_threshold=0.5, strong_signal_threshold=0.7,
                 atr_multiplier_sl=1.5, use_regime_filter=True, use_volatility_filter=True):
        self.confidence_threshold = confidence_threshold
        self.strong_signal_threshold = strong_signal_threshold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.min_adx_threshold = 15
        self.max_vol_percentile = 90
        self.correlation_threshold = 0.5

        self.technical_analyzer = TechnicalSignalGenerator()
        self.market_analyzer = MarketContextAnalyzer()
        self.logger = logging.getLogger("SignalProducer")

    def get_signal(self, model_probs, df, funding_df=None, oi_df=None):
        """
        Generate enhanced trading signal with higher confidence filtering
        and better market context integration.
        """
        if len(df) < 2:
            return {"signal_type": "NoTrade", "reason": "InsufficientData"}

        log_memory_usage(component="signal_generation")

        # Base probabilities
        P_positive = model_probs[3] + model_probs[4]
        P_negative = model_probs[0] + model_probs[1]
        P_neutral = model_probs[2]
        max_confidence = max(P_positive, P_negative)

        if max_confidence < self.confidence_threshold:
            return {
                "signal_type": "NoTrade",
                "confidence": float(max_confidence),
                "reason": "LowConfidence"
            }

        current_price = df['close'].iloc[-1]
        market_regime = self.market_analyzer.analyze_regime(df)
        volatility_score = self.market_analyzer.analyze_volatility(df)
        funding_signal = self.market_analyzer.analyze_funding(df, funding_df)
        short_term_momentum = self.market_analyzer.analyze_short_term_momentum(df)
        price_patterns = self.technical_analyzer.detect_short_term_patterns(df)

        # ATR fallback
        if 'd1_ATR_14' not in df.columns:
            atr = self.technical_analyzer.compute_atr(df).iloc[-1]
        else:
            atr = df['d1_ATR_14'].iloc[-1]

        # hist_vol fallback
        if 'hist_vol_20' in df.columns:
            hist_vol = df['hist_vol_20'].iloc[-1]
        else:
            self.logger.warning("hist_vol_20 not found, calculating on the fly")
            hist_vol = df['close'].pct_change(20).std()

        # Filter out signals in extreme volatility
        if self.use_volatility_filter and volatility_score > 0.8 and hist_vol > 0.04:
            return {
                "signal_type": "NoTrade",
                "reason": "ExtremeVolatility",
                "volatility": float(hist_vol),
                "volatility_score": float(volatility_score)
            }

        # Generate base (bullish=1, bearish=-1, neutral=0)
        base_signal = 1 if P_positive > P_negative else (-1 if P_negative > P_positive else 0)

        # Weighted combination
        combined_signal = base_signal * 0.4
        combined_signal += funding_signal * 0.15
        combined_signal += market_regime * 0.15
        combined_signal += short_term_momentum * 0.2

        oi_signal = 0
        if 'oi_price_sentiment' in df.columns:
            oi_signal = df['oi_price_sentiment'].iloc[-1]
            combined_signal += oi_signal * 0.1

        # Price patterns override
        combined_signal = self._adjust_for_candle_patterns(
            combined_signal, base_signal, price_patterns
        )

        # Trend alignment filter
        if self.use_regime_filter and abs(market_regime) > 0.7:
            if not ((market_regime > 0 and combined_signal > 0) or (market_regime < 0 and combined_signal < 0)):
                return {
                    "signal_type": "NoTrade",
                    "reason": "TrendMisalignment",
                    "regime": float(market_regime),
                    "combined_signal": float(combined_signal)
                }

        # Decide final threshold
        signal_threshold = 0.25

        # Bullish
        if combined_signal > signal_threshold:
            stop_loss_price = current_price - (self.atr_multiplier_sl * atr * (1.0 + 0.4 * volatility_score))
            tp_ratio = 2.0
            if funding_signal > 0:
                tp_ratio = 2.5
            take_profit_price = current_price + tp_ratio * (current_price - stop_loss_price)
            signal_str = "StrongBuy" if P_positive >= self.strong_signal_threshold else "Buy"

            return {
                "signal_type": signal_str,
                "confidence": float(P_positive),
                "stop_loss": round(float(stop_loss_price), 2),
                "take_profit": round(float(take_profit_price), 2),
                "regime": float(market_regime),
                "volatility": float(hist_vol),
                "funding_signal": funding_signal,
                "combined_signal": combined_signal,
                "short_term_momentum": float(short_term_momentum),
                "bullish_pattern": price_patterns['bullish_engulfing'] or price_patterns['hammer']
            }

        # Bearish
        elif combined_signal < -signal_threshold:
            stop_loss_price = current_price + (self.atr_multiplier_sl * atr * (1.0 + 0.4 * volatility_score))
            tp_ratio = 2.0
            if funding_signal < 0:
                tp_ratio = 2.5
            take_profit_price = current_price - tp_ratio * (stop_loss_price - current_price)
            signal_str = "StrongSell" if P_negative >= self.strong_signal_threshold else "Sell"

            return {
                "signal_type": signal_str,
                "confidence": float(P_negative),
                "stop_loss": round(float(stop_loss_price), 2),
                "take_profit": round(float(take_profit_price), 2),
                "regime": float(market_regime),
                "volatility": float(hist_vol),
                "funding_signal": funding_signal,
                "combined_signal": combined_signal,
                "short_term_momentum": float(short_term_momentum),
                "bearish_pattern": price_patterns['bearish_engulfing'] or price_patterns['shooting_star']
            }

        # NoTrade
        return {
            "signal_type": "NoTrade",
            "reason": "InsufficientSignal",
            "combined_signal": float(combined_signal),
            "signal_threshold": signal_threshold
        }

    ###########################################################################
    # PRIVATE / HELPER METHODS
    ###########################################################################
    def _adjust_for_candle_patterns(self, combined_signal, base_signal, patterns):
        """
        Adjust the combined signal based on bullish or bearish candlestick patterns.
        """
        if base_signal > 0:  # model bullish
            if patterns['bullish_engulfing'] or patterns['hammer']:
                combined_signal += 0.3
            elif patterns['bearish_engulfing'] or patterns['shooting_star']:
                combined_signal -= 0.4
        elif base_signal < 0:  # model bearish
            if patterns['bearish_engulfing'] or patterns['shooting_star']:
                combined_signal -= 0.3
            elif patterns['bullish_engulfing'] or patterns['hammer']:
                combined_signal += 0.4

        return combined_signal


###############################################################################
# MODULAR BACKTESTING FRAMEWORK
###############################################################################
class TradeExecutor:
    """Handles trade execution logic for backtesting."""

    def __init__(self, fixed_cost=0.001, variable_cost=0.0005, slippage=0.0005):
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        self.slippage = slippage
        self.logger = logging.getLogger("TradeExecutor")

    def execute_entry(self, signal, current_time, current_price, quantity):
        """
        Execute trade entry with costs and slippage.
        """
        direction = 'long' if signal['signal_type'].startswith('Buy') else 'short'

        slippage_amount = current_price * self.slippage
        entry_price = current_price + slippage_amount if direction == 'long' else current_price - slippage_amount
        entry_cost = self.fixed_cost + (entry_price * quantity * self.variable_cost)

        trade_entry = {
            'entry_time': current_time,
            'entry_price': entry_price,
            'direction': direction,
            'entry_signal': signal['signal_type'],
            'entry_confidence': signal.get('confidence', 0.5),
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'quantity': quantity,
            'total_entry_cost': entry_cost,
            'partial_exit_1': False,
            'partial_exit_2': False,
            'partial_exit_3': False,
            'market_regime': signal.get('regime', 0),
            'volatility_regime': signal.get('volatility', 0)
        }

        self.logger.info(
            f"{current_time} - Opening {direction} at {entry_price:.2f}, qty: {quantity:.4f}, "
            f"signal: {signal['signal_type']}, SL: {signal.get('stop_loss', 0):.2f}, "
            f"TP: {signal.get('take_profit', 0):.2f}"
        )
        return trade_entry

    def execute_exit(self, trade_entry, current_time, current_price, exit_reason,
                     exit_quantity=None, exit_confidence=None):
        """
        Execute trade exit with costs and slippage.
        """
        direction = trade_entry['direction']
        quantity = exit_quantity if exit_quantity is not None else trade_entry['quantity']
        proportion = quantity / trade_entry['quantity']
        entry_cost_alloc = proportion * trade_entry['total_entry_cost']

        slippage_amount = current_price * self.slippage
        exit_price = current_price - slippage_amount if direction == 'long' else current_price + slippage_amount
        exit_cost = self.fixed_cost + (exit_price * quantity * self.variable_cost)

        if direction == 'long':
            pnl = quantity * (exit_price - trade_entry['entry_price']) - entry_cost_alloc - exit_cost
        else:
            pnl = quantity * (trade_entry['entry_price'] - exit_price) - entry_cost_alloc - exit_cost

        trade_record = trade_entry.copy()
        trade_record.update({
            'exit_time': current_time,
            'exit_price': exit_price,
            'exit_signal': exit_reason,
            'exit_confidence': exit_confidence,
            'PnL': pnl,
            'quantity': quantity
        })

        self.logger.info(
            f"{current_time} - Closing {direction} at {exit_price:.2f}, "
            f"PnL: {pnl:.2f}, Reason: {exit_reason}"
        )
        return trade_record, pnl


class EnhancedStrategyBacktester:
    """
    Enhanced backtesting framework with memory optimization and modular design.
    """

    def __init__(self, data_df, preparer, modeler, signal_producer, risk_manager,
                 oi_df=None, funding_df=None,
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
        self.walk_forward_steps = walk_forward_steps
        self.monte_carlo_sims = monte_carlo_sims
        self.logger = self._setup_logger()
        self.oi_df = oi_df
        self.funding_df = funding_df

        self.trade_executor = TradeExecutor(
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            slippage=slippage
        )

        self._adjust_window_sizes()

    def _setup_logger(self):
        logger = logging.getLogger("EnhancedBacktester")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            f"EnhancedTrainingResults/BacktestLog/backtest_log_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _adjust_window_sizes(self):
        df_len = len(self.data_df)
        min_required = 50
        if df_len < min_required:
            self.logger.warning(f"Insufficient data ({df_len} rows). Need at least {min_required} rows.")
            return False

        if df_len < 500:
            orig_train = self.train_window_size
            orig_test = self.test_window_size
            self.train_window_size = max(int(df_len * 0.5), 30)
            self.test_window_size = max(int(df_len * 0.3), 20)
            self.logger.info(
                f"Adjusted window sizes for small dataset: train_window={orig_train}->{self.train_window_size}, "
                f"test_window={orig_test}->{self.test_window_size}"
            )
        return True

    def walk_forward_backtest(self):
        """
        Enhanced walk-forward backtest with memory optimization
        and improved error handling.
        """
        log_memory_usage(component="backtest_start")
        start_idx = 0
        df_len = len(self.data_df)
        iteration = 0

        results_dir = "EnhancedTrainingResults/Trades"
        os.makedirs(results_dir, exist_ok=True)
        results_path = f"{results_dir}/trades_{datetime.now():%Y%m%d_%H%M%S}.csv"

        with open(results_path, 'w') as f:
            f.write("iteration,entry_time,exit_time,direction,entry_price,exit_price,quantity,PnL,"
                    "entry_signal,exit_signal,regime,stop_loss,take_profit\n")

        performance_by_iteration = []
        step_size = max(self.test_window_size // 2, 100)
        all_results = []
        max_iterations = min(self.walk_forward_steps if self.walk_forward_steps > 0 else 99, 8)

        while start_idx + self.train_window_size + self.test_window_size <= df_len and iteration < max_iterations:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration} of walk-forward backtest")
            log_memory_usage(component=f"backtest_iteration_{iteration}_start")

            train_end = start_idx + self.train_window_size
            test_end = min(train_end + self.test_window_size, df_len)
            df_train = self.data_df.iloc[start_idx:train_end].copy()
            df_test = self.data_df.iloc[train_end:test_end].copy()

            for df_local in [df_train, df_test]:
                for col in df_local.select_dtypes(include=['float64']).columns:
                    df_local[col] = df_local[col].astype(np.float32)

            regime = self._detect_regime(df_train)
            self.logger.info(f"Detected market regime: {regime}")
            memory_watchdog(threshold_gb=20, component=f"backtest_iteration_{iteration}_before_prep")

            try:
                X_train, y_train, X_val, y_val, df_val, fwd_returns_val = self.preparer.prepare_data(df_train)
            except Exception as e:
                self.logger.error(f"Error in data preparation: {e}", exc_info=True)
                start_idx += step_size
                continue

            if len(X_train) == 0:
                self.logger.warning(f"Insufficient training data in iteration {iteration}")
                start_idx += step_size
                continue

            memory_watchdog(threshold_gb=20, component=f"backtest_iteration_{iteration}_after_prep")

            try:
                y_train_flat = np.argmax(y_train, axis=1)
                unique_classes = np.unique(y_train_flat)
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_flat)
                class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)}
                for cls in range(5):
                    if cls not in class_weight_dict:
                        avg_weight = np.mean(list(class_weight_dict.values())) if class_weight_dict else 1.0
                        class_weight_dict[cls] = avg_weight

                if regime == 'trending':
                    if 0 in class_weight_dict:
                        class_weight_dict[0] *= 1.75
                    if 4 in class_weight_dict:
                        class_weight_dict[4] *= 1.75
                elif regime == 'ranging':
                    if 1 in class_weight_dict:
                        class_weight_dict[1] *= 1.5
                    if 3 in class_weight_dict:
                        class_weight_dict[3] *= 1.5
            except Exception as e:
                self.logger.warning(f"Error computing class weights: {e}")
                class_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

            batch_size = 64
            epochs = 10
            memory_watchdog(threshold_gb=20, component=f"backtest_iteration_{iteration}_before_training")

            try:
                self.logger.info(f"Training model for iteration {iteration}")
                self.modeler.tune_and_train(
                    iteration, X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                    epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict
                )
            except Exception as e:
                self.logger.error(f"Error training model: {e}", exc_info=True)
                start_idx += step_size
                continue

            memory_watchdog(threshold_gb=20, component=f"backtest_iteration_{iteration}_after_training", force_cleanup=True)
            del X_train, y_train, X_val, y_val, df_val, fwd_returns_val
            collect()

            memory_watchdog(threshold_gb=20, component=f"backtest_iteration_{iteration}_before_simulation")

            try:
                self.logger.info(f"Simulating trading for iteration {iteration}")
                test_eq, test_trades = self._simulate_test(df_test, iteration, regime)
            except Exception as e:
                self.logger.error(f"Error in trade simulation: {e}", exc_info=True)
                start_idx += step_size
                continue

            self._save_trades_to_file(test_trades, results_path)

            iter_result = {
                "iteration": iteration,
                "train_start": start_idx,
                "train_end": train_end,
                "test_end": test_end,
                "final_equity": test_eq,
                "regime": regime
            }
            all_results.append(iter_result)

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

            checkpoint_path = f"{results_dir}/checkpoint_iter_{iteration}.json"
            with open(checkpoint_path, 'w') as f:
                import json
                json.dump(iter_result, f)

            del test_trades, df_train, df_test
            collect()
            memory_watchdog(threshold_gb=20, component=f"backtest_iteration_{iteration}_end", force_cleanup=True)
            start_idx += int(step_size)

        self._analyze_period_performance(performance_by_iteration)
        perf_summary_path = f"{results_dir}/performance_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(perf_summary_path, 'w') as f:
            import json
            json.dump([
                {
                    **{k: v for k, v in p.items() if k != "metrics"},
                    "metrics_summary": {
                        metric: val for metric, val in p.get("metrics", {}).items()
                        if metric in ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "return"]
                    }
                }
                for p in performance_by_iteration
            ], f, default=str)

        memory_watchdog(threshold_gb=20, component="backtest_end", force_cleanup=True)
        if not all_results:
            self.logger.warning("No backtest iterations were completed")
            return pd.DataFrame()

        return pd.DataFrame(all_results)

    def _save_trades_to_file(self, trades, filepath):
        if not trades:
            return
        with open(filepath, 'a') as f:
            for trade in trades:
                line = (
                    f"{trade.get('iteration', 0)},"
                    f"{trade.get('entry_time', '')},"
                    f"{trade.get('exit_time', '')},"
                    f"{trade.get('direction', '')},"
                    f"{trade.get('entry_price', 0)},"
                    f"{trade.get('exit_price', 0)},"
                    f"{trade.get('quantity', 0)},"
                    f"{trade.get('PnL', 0)},"
                    f"{trade.get('entry_signal', '').replace(',', ';')},"
                    f"{trade.get('exit_signal', '').replace(',', ';')},"
                    f"{trade.get('regime', '').replace(',', ';')},"
                    f"{trade.get('stop_loss', 0)},"
                    f"{trade.get('take_profit', 0)}\n"
                )
                f.write(line)
        self.logger.info(f"Saved {len(trades)} trades to {filepath}")

    def _detect_regime(self, df):
        if len(df) < 100:
            return "unknown"

        if 'market_regime' in df.columns and 'volatility_regime' in df.columns:
            market_regime = df['market_regime'].iloc[-20:].mean()
            volatility_regime = df['volatility_regime'].iloc[-20:].mean()
            if volatility_regime > 0.5:
                return "volatile"
            elif abs(market_regime) > 0.5:
                return "trending"
            else:
                return "ranging"

        close = df['close'].values
        returns = np.diff(close) / close[:-1] if len(close) > 1 else []
        volatility = np.std(returns[-50:]) * np.sqrt(252) if len(returns) >= 50 else 0

        x = np.arange(min(50, len(close)))
        if len(x) < 2:
            return "unknown"
        slope, _, r_value, _, _ = linregress(x, close[-len(x):])
        trend_strength = abs(r_value)
        normalized_slope = slope / close[-1] * 100 if close[-1] != 0 else 0

        if volatility > 0.8:
            return "volatile"
        elif trend_strength > 0.7 and abs(normalized_slope) > 0.1:
            return "trending"
        else:
            return "ranging"

    def _calculate_performance_metrics(self, trades, final_equity):
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

        wins = [t for t in trades if t['PnL'] > 0]
        losses = [t for t in trades if t['PnL'] <= 0]
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        total_profit = sum(t['PnL'] for t in wins)
        total_loss = abs(sum(t['PnL'] for t in losses)) if losses else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        initial_capital = self.risk_manager.initial_capital
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['PnL'])

        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

        sharpe = 0
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0

        peak = equity_curve[0]
        drawdowns = []
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            drawdowns.append(dd)
        max_drawdown = max(drawdowns)

        avg_trade = sum(t['PnL'] for t in trades) / len(trades) if trades else 0
        total_return = (final_equity - initial_capital) / initial_capital * 100

        avg_win = sum(t['PnL'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['PnL'] for t in losses) / len(losses) if losses else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        holding_times = []
        for trade in trades:
            entry = pd.to_datetime(trade['entry_time'])
            exit_ = pd.to_datetime(trade['exit_time'])
            holding_times.append((exit_ - entry).total_seconds() / 3600)
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
        if not performance_by_iteration:
            return

        regime_groups = {}
        for perf in performance_by_iteration:
            reg = perf['regime']
            regime_groups.setdefault(reg, []).append(perf)

        regime_metrics = {}
        for r, performances in regime_groups.items():
            avg_metrics = {}
            for metric in performances[0]['metrics']:
                values = [p['metrics'][metric] for p in performances]
                avg_metrics[metric] = sum(values) / len(values)
            regime_metrics[r] = avg_metrics

        self.logger.info("\n=== Regime Performance Analysis ===")
        for regime, metrics in regime_metrics.items():
            self.logger.info(f"\nRegime: {regime}")
            for metric, value in metrics.items():
                self.logger.info(f"Avg {metric}: {value}")

    def _simulate_test(self, df_test, iteration, regime="unknown"):
        """
        Simulate trading on test data with improved memory management.
        """
        log_memory_usage(component=f"simulate_test_{iteration}_start")
        try:
            X_test, y_test, df_labeled, _ = self.preparer.prepare_test_data(df_test)
        except Exception as e:
            self.logger.error(f"Error preparing test data: {e}")
            return self.risk_manager.initial_capital, []

        if len(X_test) == 0:
            self.logger.warning(f"Insufficient test data in iteration {iteration}")
            return self.risk_manager.initial_capital, []

        memory_watchdog(threshold_gb=15, component=f"simulate_test_{iteration}_after_prep")

        try:
            batch_size = 32
            predictions = []
            for i in range(0, len(X_test), batch_size):
                end_idx = min(i + batch_size, len(X_test))
                batch_preds = self.modeler.predict_signals(X_test[i:end_idx])
                predictions.append(batch_preds)
                if i % 100 == 0:
                    memory_watchdog(threshold_gb=15, component=f"simulate_test_{iteration}_prediction_batch")

            preds = np.vstack(predictions) if len(predictions) > 0 else np.array([])
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return self.risk_manager.initial_capital, []

        memory_watchdog(threshold_gb=15, component=f"simulate_test_{iteration}_after_predictions")

        self.risk_manager.current_capital = self.risk_manager.initial_capital
        self.risk_manager.open_positions = []
        self.risk_manager.trade_history = []

        equity_curve = [self.risk_manager.current_capital]
        trades = []
        sequence_length = self.preparer.sequence_length

        position = 0
        trade_entry = None

        for i in range(len(preds)):
            if i % 100 == 0:
                memory_watchdog(threshold_gb=15, component=f"simulate_test_{iteration}_simulation")

            current_row = i + sequence_length - 1
            if current_row >= len(df_test):
                break

            current_time = df_test.index[current_row]
            current_price = df_test['close'].iloc[current_row]
            model_probs = preds[i]
            current_df_slice = df_test.iloc[:current_row + 1]

            current_funding_df = None
            if self.funding_df is not None:
                current_funding_df = self.funding_df[self.funding_df.index <= current_time]
            current_oi_df = None
            if self.oi_df is not None:
                current_oi_df = self.oi_df[self.oi_df.index <= current_time]

            try:
                signal = self.signal_producer.get_signal(
                    model_probs,
                    current_df_slice,
                    funding_df=current_funding_df,
                    oi_df=current_oi_df
                )
            except Exception as e:
                self.logger.error(f"Error generating signal at {current_time}: {e}")
                signal = {"signal_type": "NoTrade", "reason": f"SignalError: {str(e)}"}

            # Manage open position
            if position != 0:
                try:
                    atr_series = self.signal_producer.technical_analyzer.compute_atr(df_test)
                    current_atr = atr_series.iloc[current_row] if current_row < len(atr_series) else 0
                except Exception as e:
                    self.logger.warning(f"Error computing ATR: {e}, using default value")
                    current_atr = current_price * 0.01

                current_funding_rate = 0
                if hasattr(current_funding_df, 'fundingRate') and not current_funding_df.empty:
                    current_funding_rate = current_funding_df['fundingRate'].iloc[-1]

                try:
                    exit_decision = self.risk_manager.dynamic_exit_strategy(
                        trade_entry, current_price, current_atr, current_funding_rate
                    )
                except Exception as e:
                    self.logger.warning(f"Error in exit strategy: {e}")
                    exit_decision = {"update_stop": False}

                if exit_decision.get('update_stop'):
                    trade_entry['stop_loss'] = exit_decision['new_stop']
                    self.logger.info(f"{current_time} - Updated stop to {trade_entry['stop_loss']:.2f}")

                if exit_decision.get('partial_exit'):
                    exit_ratio = exit_decision.get('exit_ratio', 0.5)
                    exit_reason = exit_decision.get('reason', 'PartialExit')
                    close_quantity = trade_entry['quantity'] * exit_ratio
                    try:
                        partial_trade, pnl = self.trade_executor.execute_exit(
                            trade_entry, current_time, current_price, exit_reason,
                            exit_quantity=close_quantity
                        )
                        trade_entry['quantity'] -= close_quantity

                        if exit_reason == "FirstTarget":
                            trade_entry['partial_exit_1'] = True
                        elif exit_reason == "SecondTarget":
                            trade_entry['partial_exit_2'] = True
                        elif exit_reason == "ThirdTarget":
                            trade_entry['partial_exit_3'] = True

                        partial_trade['iteration'] = iteration
                        partial_trade['regime'] = regime
                        trades.append(partial_trade)
                        self.risk_manager.current_capital += pnl

                        self.risk_manager.trade_history.append({
                            'entry_time': trade_entry['entry_time'],
                            'exit_time': current_time,
                            'direction': trade_entry['direction'],
                            'pnl': pnl,
                            'exit_reason': exit_reason
                        })
                    except Exception as e:
                        self.logger.error(f"Error executing partial exit: {e}")

                hit_stop = (
                    (position > 0 and current_price <= trade_entry['stop_loss']) or
                    (position < 0 and current_price >= trade_entry['stop_loss'])
                )
                signal_reversal = (
                    (position > 0 and "Sell" in signal['signal_type']) or
                    (position < 0 and "Buy" in signal['signal_type'])
                )
                if hit_stop or signal_reversal:
                    exit_reason = signal['signal_type'] if signal_reversal else "StopLoss"
                    exit_confidence = signal.get('confidence', None) if signal_reversal else None
                    try:
                        trade_record, pnl = self.trade_executor.execute_exit(
                            trade_entry, current_time, current_price, exit_reason,
                            exit_confidence=exit_confidence
                        )
                        trade_record['iteration'] = iteration
                        trade_record['regime'] = regime
                        trades.append(trade_record)
                        self.risk_manager.current_capital += pnl

                        self.risk_manager.trade_history.append({
                            'entry_time': trade_entry['entry_time'],
                            'exit_time': current_time,
                            'direction': trade_entry['direction'],
                            'pnl': pnl,
                            'exit_reason': exit_reason
                        })

                        position = 0
                        trade_entry = None
                    except Exception as e:
                        self.logger.error(f"Error executing full exit: {e}")

            # Check for entry
            if position == 0 and ("Buy" in signal['signal_type'] or "Sell" in signal['signal_type']):
                direction = 'long' if "Buy" in signal['signal_type'] else 'short'
                try:
                    can_add, adjusted_risk = self.risk_manager.check_correlation_risk({
                        'direction': direction,
                        'risk_amount': self.risk_manager.current_capital * self.risk_manager.max_risk_per_trade
                    })
                except Exception as e:
                    self.logger.warning(f"Error checking correlation risk: {e}")
                    can_add = True
                    adjusted_risk = self.risk_manager.max_risk_per_trade

                if not can_add:
                    self.logger.info(f"{current_time} - Signal rejected due to correlation risk")
                    continue

                stop_loss = signal.get('stop_loss', 0)
                if stop_loss == 0:
                    try:
                        atr = self.signal_producer.technical_analyzer.compute_atr(df_test).iloc[current_row]
                        if np.isnan(atr) or atr <= 0:
                            atr = current_price * 0.01
                    except Exception:
                        atr = current_price * 0.01
                    distance = self.signal_producer.atr_multiplier_sl * atr
                    stop_loss = current_price - distance if direction == 'long' else current_price + distance

                try:
                    volatility_regime = signal.get('volatility', 0)
                    if 'volatility_regime' in df_test.columns:
                        volatility_regime = df_test['volatility_regime'].iloc[current_row]

                    current_funding_rate = 0
                    if hasattr(current_funding_df, 'fundingRate') and not current_funding_df.empty:
                        current_funding_rate = current_funding_df['fundingRate'].iloc[-1]

                    quantity = self.risk_manager.calculate_position_size(
                        signal, current_price, stop_loss, volatility_regime, current_funding_rate
                    )
                except Exception as e:
                    self.logger.warning(f"Error calculating position size: {e}")
                    risk_amount = self.risk_manager.current_capital * self.risk_manager.max_risk_per_trade * 0.8
                    risk_per_unit = abs(current_price - stop_loss)
                    quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                if quantity <= 0:
                    self.logger.info(f"{current_time} - Zero position size calculated")
                    continue

                try:
                    trade_entry = self.trade_executor.execute_entry(signal, current_time, current_price, quantity)
                    position = 1 if direction == 'long' else -1
                    trade_entry['iteration'] = iteration
                    trade_entry['regime'] = regime
                    self.risk_manager.open_positions.append({
                        'direction': direction,
                        'entry_time': current_time,
                        'entry_price': trade_entry['entry_price'],
                        'stop_loss': stop_loss,
                        'take_profit': trade_entry.get('take_profit', 0),
                        'quantity': quantity,
                        'risk_amount': quantity * abs(trade_entry['entry_price'] - stop_loss)
                    })
                except Exception as e:
                    self.logger.error(f"Error executing entry: {e}")
                    continue

            equity_curve.append(self.risk_manager.current_capital)

        if position != 0 and trade_entry is not None:
            exit_time = df_test.index[-1]
            exit_reason = "EndOfTest"
            try:
                trade_record, pnl = self.trade_executor.execute_exit(
                    trade_entry, exit_time, df_test['close'].iloc[-1], exit_reason
                )
                trade_record['iteration'] = iteration
                trade_record['regime'] = regime
                trades.append(trade_record)
                self.risk_manager.current_capital += pnl

                self.risk_manager.trade_history.append({
                    'entry_time': trade_entry['entry_time'],
                    'exit_time': exit_time,
                    'direction': trade_entry['direction'],
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
            except Exception as e:
                self.logger.error(f"Error executing final exit: {e}")

        self.risk_manager.update_performance_stats()
        memory_watchdog(threshold_gb=15, component=f"simulate_test_{iteration}_end")
        return self.risk_manager.current_capital, trades
