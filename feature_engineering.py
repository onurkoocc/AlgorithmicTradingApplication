import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import gc
from config import memory_watchdog, log_memory_usage

###############################################################################
# ENHANCED FEATURE ENGINEERING
###############################################################################
class EnhancedCryptoFeatureEngineer:
    def __init__(self, feature_scaling=False):
        self.feature_scaling = feature_scaling
        # Daily parameters
        self.ma_periods_daily = [10, 20, 50]
        self.rsi_period_daily = 14
        self.macd_fast_daily = 12
        self.macd_slow_daily = 26
        self.macd_signal_daily = 9
        self.bb_period_daily = 20
        self.bb_stddev_daily = 2
        self.atr_period_daily = 14
        self.mfi_period_daily = 14
        self.cmf_period_daily = 21
        # 4H parameters
        self.ma_periods_4h = [20, 50, 100, 200]
        self.rsi_period_4h = 14
        self.macd_fast_4h = 12
        self.macd_slow_4h = 26
        self.macd_signal_4h = 9
        self.mfi_period_4h = 14
        self.adx_period_4h = 14
        # 30m parameters
        self.cmf_period_30m = 20
        self.obv_ma_period_30m = 10
        self.mfi_period_30m = 14
        self.force_ema_span_30m = 2
        self.vwap_period_30m = 20
        # Enhanced parameters
        self.regime_window = 20
        self.volume_zones_lookback = 50
        self.swing_threshold = 0.5
        # Setup logger
        self.logger = logging.getLogger("FeatureEngineer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def process_data_3way(self, df_30m, df_4h, df_daily, oi_df=None, funding_df=None):
        """Process data with more aggressive memory management and better handling of derived timeframes"""
        # Log memory at start
        log_memory_usage()

        # Ensure dataframes have consistent dtypes
        # Convert DataFrames to float32 to reduce memory usage
        for df in [df_30m, df_4h, df_daily]:
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)

        # Ensure all dataframes have datetime index with no timezone for consistent alignment
        for df in [df_30m, df_4h, df_daily]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # Remove timezone info to avoid alignment issues
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        # Get base features with reduced memory footprint
        feat_30m = self._compute_indicators_30m(df_30m).add_prefix('m30_')
        feat_30m[['open', 'high', 'low', 'close', 'volume']] = df_30m[['open', 'high', 'low', 'close', 'volume']]

        # Clear unused DataFrames
        del df_30m
        gc.collect()

        feat_4h = self._compute_indicators_4h(df_4h).add_prefix('h4_')
        feat_4h_ff = self._align_timeframes(feat_4h, feat_30m.index)
        del df_4h, feat_4h
        gc.collect()

        feat_daily = self._compute_indicators_daily(df_daily).add_prefix('d1_')
        feat_daily_ff = self._align_timeframes(feat_daily, feat_30m.index)
        del df_daily, feat_daily
        gc.collect()

        # Use a more memory-efficient concat
        combined = pd.concat([feat_30m, feat_4h_ff, feat_daily_ff], axis=1, copy=False)
        critical_columns = ['open', 'high', 'low', 'close', 'volume']
        combined.dropna(subset=critical_columns, inplace=True)
        print(f"After dropna: {len(combined)} rows remaining")

        # Check if we have data after dropping NA values
        if combined.empty:
            print("WARNING: No data available after combining and dropping NA values")
            # Return a minimal DataFrame with expected columns
            return pd.DataFrame({
                'open': [], 'high': [], 'low': [], 'close': [], 'volume': [],
                'market_regime': [], 'volatility_regime': [], 'trend_strength': [],
                'swing_high': [], 'swing_low': []
            })

        # Clear unused DataFrames
        del feat_30m, feat_4h_ff, feat_daily_ff
        gc.collect()
        memory_watchdog(threshold_gb=20)

        # Add enhanced features with improved memory handling
        # Add market regime detection - check for minimum required data
        if len(combined) > 20:  # Need at least 20 rows for meaningful regime calculation
            try:
                combined['market_regime'] = self._compute_market_regime(combined)
            except Exception as e:
                print(f"WARNING: Error computing market regime: {e}")
                combined['market_regime'] = 0
        else:
            combined['market_regime'] = 0

        memory_watchdog(threshold_gb=20)

        # Add funding rate features if available
        if funding_df is not None and not funding_df.empty:
            try:
                funding_features = self._compute_funding_rate_features(combined, funding_df)
                combined = pd.concat([combined, funding_features], axis=1)
                memory_watchdog(threshold_gb=20)
            except Exception as e:
                print(f"WARNING: Error computing funding rate features: {e}")

        # Add open interest features if available
        if oi_df is not None and not oi_df.empty:
            try:
                oi_features = self._compute_open_interest_features(combined, oi_df)
                combined = pd.concat([combined, oi_features], axis=1)
                memory_watchdog(threshold_gb=20)
            except Exception as e:
                print(f"WARNING: Error computing open interest features: {e}")

        # Add volume profile features
        try:
            combined['volume_zone'] = self._compute_volume_zones(combined)
        except Exception as e:
            print(f"WARNING: Error computing volume zones: {e}")
            combined['volume_zone'] = 0

        memory_watchdog(threshold_gb=12)

        # Add swing detection
        try:
            combined['swing_high'] = self._detect_swing_highs(combined, self.swing_threshold)
            combined['swing_low'] = self._detect_swing_lows(combined, self.swing_threshold)
        except Exception as e:
            print(f"WARNING: Error detecting swings: {e}")
            combined['swing_high'] = 0
            combined['swing_low'] = 0

        memory_watchdog(threshold_gb=12)

        # Add volatility regime
        try:
            combined['volatility_regime'] = self._compute_volatility_regime(combined)
        except Exception as e:
            print(f"WARNING: Error computing volatility regime: {e}")
            combined['volatility_regime'] = 0

        # Add mean reversion potential
        try:
            combined['mean_reversion_potential'] = self._compute_mean_reversion(combined)
        except Exception as e:
            print(f"WARNING: Error computing mean reversion: {e}")
            combined['mean_reversion_potential'] = 0

        # Add trend strength
        try:
            combined['trend_strength'] = self._compute_trend_strength(combined)
        except Exception as e:
            print(f"WARNING: Error computing trend strength: {e}")
            combined['trend_strength'] = 0

        # Feature selection with error handling
        try:
            top_features = self.select_top_features(combined, n_top=50)  # Reduced from 65 to 50
        except Exception as e:
            print(f"WARNING: Error during feature selection: {e}")
            # Just use all available numeric columns as features
            numeric_cols = combined.select_dtypes(include=np.number).columns.tolist()
            # Exclude the columns we're going to explicitly include below
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'market_regime',
                            'volatility_regime', 'trend_strength', 'swing_high', 'swing_low']
            top_features = [col for col in numeric_cols if col not in exclude_cols][:50]

        # Make sure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'market_regime', 'volatility_regime',
                            'trend_strength', 'swing_high', 'swing_low']
        for col in required_columns:
            if col not in combined.columns:
                print(f"WARNING: Required column {col} not found, adding zero-filled column")
                combined[col] = 0

        # Select only the columns we need
        selected_columns = required_columns + list(top_features)

        # Ensure the columns actually exist in the DataFrame
        existing_columns = [col for col in selected_columns if col in combined.columns]

        if not existing_columns:
            print("ERROR: No valid columns found after selection")
            # Create a minimal DataFrame with expected structure
            return pd.DataFrame(columns=required_columns)

        combined = combined[existing_columns]

        if self.feature_scaling:
            combined = self._scale_features(combined)

        print("Preview of combined features:\n", combined.head())
        log_memory_usage()
        return combined

    def _align_timeframes(self, higher_tf_data, target_index):
        """Align higher timeframe data to lower timeframe index"""
        if higher_tf_data.empty or len(target_index) == 0:
            return pd.DataFrame(index=target_index)

        # Create a DataFrame with the target index
        aligned_data = pd.DataFrame(index=target_index)

        # For each higher timeframe row
        for idx, row in higher_tf_data.iterrows():
            # Identify all target timeframe indices that fall within this candle's time range
            if idx == higher_tf_data.index[-1]:
                # For the last candle, include all remaining target indices
                mask = (target_index >= idx)
            else:
                # For other candles, find target indices between this candle and the next
                next_idx = higher_tf_data.index[higher_tf_data.index.get_loc(idx) + 1]
                mask = (target_index >= idx) & (target_index < next_idx)

            # Set values for all columns
            for col in higher_tf_data.columns:
                if col not in aligned_data:
                    aligned_data[col] = np.nan
                aligned_data.loc[mask, col] = row[col]

        # Fill any remaining NaN values with ffill instead of method='ffill'
        aligned_data = aligned_data.ffill()

        return aligned_data

    def process_data_in_chunks(self, df_30m, df_4h, df_daily, chunk_size=2000):
        """Process data in chunks with better handling of derived timeframes"""
        results = []
        for i in range(0, len(df_30m), chunk_size):
            log_memory_usage()

            # Extract chunk
            end_idx = min(i + chunk_size, len(df_30m))
            chunk_30m = df_30m.iloc[i:end_idx]

            # Find corresponding indices in derived timeframes
            start_time = chunk_30m.index[0]
            end_time = chunk_30m.index[-1]

            # For derived timeframes, we need to be a bit more careful with the selection
            # We need to include the last candle before the start time to ensure proper forward filling
            chunk_4h = df_4h[(df_4h.index <= end_time)]
            if not chunk_4h.empty:
                # Make sure we have at least one candle before start_time for proper alignment
                if chunk_4h.index[0] > start_time and len(df_4h) > len(chunk_4h):
                    # Find the last candle before our chunk starts
                    prev_idx = df_4h[df_4h.index < start_time].index[-1]
                    # Add it to our chunk
                    chunk_4h = pd.concat([df_4h.loc[[prev_idx]], chunk_4h])

            chunk_daily = df_daily[(df_daily.index <= end_time)]
            if not chunk_daily.empty:
                # Same logic for daily data
                if chunk_daily.index[0] > start_time and len(df_daily) > len(chunk_daily):
                    prev_idx = df_daily[df_daily.index < start_time].index[-1]
                    chunk_daily = pd.concat([df_daily.loc[[prev_idx]], chunk_daily])

            # Process chunk
            chunk_features = self.process_data_3way(chunk_30m, chunk_4h, chunk_daily)
            results.append(chunk_features)

            # Clear memory
            del chunk_30m, chunk_4h, chunk_daily, chunk_features
            memory_watchdog(threshold_gb=15)

        # Combine results (carefully to avoid memory spike)
        log_memory_usage()
        combined = pd.concat(results, copy=False)

        del results
        gc.collect()

        return combined

    def select_top_features(self, df, n_top=65):
        """Select top features using RandomForest feature importance with consistent X and y shapes"""
        # Ensure n_top doesn't exceed the number of available features
        n_features = len(df.columns) - 10  # Subtract the reserved columns
        n_top = min(n_top, n_features) if n_features > 0 else 1

        # Set a reasonable sample size based on available data
        sample_size = min(5000, len(df))
        horizon = 48  # Must match the horizon in _create_labels_for_selection

        # Handle case where sample size is too small
        if sample_size <= horizon:
            self.logger.warning(
                f"Sample size ({sample_size}) too small for horizon ({horizon}). Using all data and reduced horizon.")
            # Use all available data
            sample_size = len(df)
            # Reduce horizon if needed to ensure we have enough data
            horizon = min(horizon, max(1, int(sample_size / 10)))

            # If we still don't have enough data, return a default set of features
            if sample_size <= horizon:
                self.logger.warning("Insufficient data for feature selection. Using default features.")
                # Return all available features since we don't have enough data to select intelligently
                columns_to_exclude = ['open', 'high', 'low', 'close', 'volume', 'market_regime',
                                      'volatility_regime', 'trend_strength', 'swing_high', 'swing_low']
                available_features = [col for col in df.columns if col not in columns_to_exclude]
                return available_features[:n_top] if available_features else ['close']

        # Define columns to exclude from feature selection
        columns_to_exclude = ['open', 'high', 'low', 'close', 'volume', 'market_regime',
                              'volatility_regime', 'trend_strength', 'swing_high', 'swing_low']

        # Check if we have any columns left after exclusion
        available_features = [col for col in df.columns if col not in columns_to_exclude]

        if not available_features:
            self.logger.warning("No features available for selection after excluding reserved columns.")
            return ['close']  # Return at least one feature

        try:
            # Generate labels first (this accounts for the horizon)
            y = self._create_labels_for_selection(df.iloc[:sample_size], horizon)

            # Now create features that match the label length exactly
            X = df.iloc[:(sample_size - horizon)].drop(columns=columns_to_exclude, errors='ignore')

            # Double-check that X and y have the same length
            if len(X) != len(y):
                self.logger.warning(
                    f"X and y have different lengths: {len(X)} vs {len(y)}. Using simple feature selection.")
                # If they still don't match, return available features
                return available_features[:n_top] if available_features else ['close']

            # Check if X has any columns
            if X.empty or len(X.columns) == 0:
                self.logger.warning("No features available for selection after dropping reserved columns.")
                return available_features[:n_top] if available_features else ['close']

            # Check if y has enough samples
            if len(y) < 10:
                self.logger.warning("Not enough label samples for feature selection.")
                return available_features[:n_top] if available_features else ['close']

            # Train RandomForest for feature selection
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = X.columns[indices[:n_top]]
            return top_features

        except Exception as e:
            self.logger.error(f"RandomForest feature selection failed: {e}")
            # If feature selection fails, return available features
            return available_features[:n_top] if available_features else ['close']

    def _create_labels_for_selection(self, df, horizon=48):
        """Create classification labels for feature selection with length matching"""
        if len(df) <= horizon:
            raise ValueError(f"DataFrame length ({len(df)}) must be greater than horizon ({horizon})")

        # Use 'close' column for label creation
        if 'close' not in df.columns:
            raise ValueError("'close' column required for label creation")

        # We need to account for the horizon in our calculations
        # First, create a Series of appropriate length
        labels = pd.Series(index=df.index[:-horizon], dtype=np.int8)

        # Calculate forward returns safely
        price = df['close'].values

        # For each valid position, calculate the forward return
        for i in range(len(price) - horizon):
            if np.isnan(price[i]) or np.isnan(price[i + horizon]):
                labels.iloc[i] = 2  # Neutral class for NaN values
            else:
                # Calculate return
                fwd_return = price[i + horizon] / price[i] - 1

                # Create labels based on return thresholds
                if fwd_return < -0.05:
                    labels.iloc[i] = 0  # Strongly bearish
                elif fwd_return < -0.01:
                    labels.iloc[i] = 1  # Moderately bearish
                elif fwd_return < 0.01:
                    labels.iloc[i] = 2  # Neutral
                elif fwd_return < 0.05:
                    labels.iloc[i] = 3  # Moderately bullish
                else:
                    labels.iloc[i] = 4  # Strongly bullish

        # Check for NaN values and fill with neutral class
        labels = labels.fillna(2).astype(np.int8)

        return labels.values

    def _compute_indicators_30m(self, df):
        """Compute technical indicators for 30-minute data"""
        out = pd.DataFrame(index=df.index)
        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']
        # Bollinger Bands
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + 2 * std
        out['BB_lower'] = mid - 2 * std
        out['BB_width'] = out['BB_upper'] - out['BB_lower']
        # Historical Volatility
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std()
        # Chaikin Money Flow
        multiplier = np.where(df['high'] != df['low'],
                              ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']),
                              0).astype(np.float32)
        money_flow_volume = multiplier * df['volume']
        out[f'CMF_{self.cmf_period_30m}'] = (
                money_flow_volume.rolling(self.cmf_period_30m).sum() / df['volume'].rolling(
            self.cmf_period_30m).sum()).astype(np.float32)
        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        out['OBV'] = obv
        out[f'OBV_SMA_{self.obv_ma_period_30m}'] = obv.rolling(self.obv_ma_period_30m).mean()
        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        price_diff = typical_price.diff()
        pos_sum = pd.Series(np.where(price_diff > 0, raw_money_flow, 0), index=df.index).rolling(
            self.mfi_period_30m).sum()
        neg_sum = pd.Series(np.where(price_diff < 0, raw_money_flow, 0), index=df.index).rolling(
            self.mfi_period_30m).sum()
        money_flow_ratio = pos_sum / neg_sum.replace(0, np.nan)
        out[f'MFI_{self.mfi_period_30m}'] = (100 - (100 / (1 + money_flow_ratio))).astype(np.float32)
        # Force Index
        force_index_1 = (df['close'] - df['close'].shift(1)) * df['volume']
        out[f'ForceIndex_EMA{self.force_ema_span_30m}'] = force_index_1.ewm(span=self.force_ema_span_30m,
                                                                            adjust=False).mean().astype(np.float32)
        # VWAP
        out[f'VWAP_{self.vwap_period_30m}'] = (
                (df['close'] * df['volume']).rolling(self.vwap_period_30m).sum() / df['volume'].rolling(
            self.vwap_period_30m).sum()).astype(np.float32)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    def _compute_indicators_4h(self, df):
        """Compute technical indicators for 4-hour data"""
        out = pd.DataFrame(index=df.index)
        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']
        # Bollinger Bands
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + 2 * std
        out['BB_lower'] = mid - 2 * std
        out['BB_width'] = out['BB_upper'] - out['BB_lower']
        # Historical Volatility
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std()
        # MAs
        for period in self.ma_periods_4h:
            out[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            out[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(self.rsi_period_4h).mean()
        avg_loss = down.rolling(self.rsi_period_4h).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f'RSI_{self.rsi_period_4h}'] = 100 - (100 / (1 + rs))
        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast_4h, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow_4h, adjust=False).mean()
        macd_ = ema_fast - ema_slow
        macd_signal_ = macd_.ewm(span=self.macd_signal_4h, adjust=False).mean()
        out['MACD'] = macd_
        out['MACD_signal'] = macd_signal_
        out['MACD_hist'] = macd_ - macd_signal_
        # OBV
        out['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum().fillna(0)
        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        tp_diff = typical_price.diff()
        pos_flow = raw_money_flow.where(tp_diff > 0, 0)
        neg_sum = raw_money_flow.where(tp_diff < 0, 0)
        out[f'MFI_{self.mfi_period_4h}'] = (100 - (100 / (1 + (
                pos_flow.rolling(self.mfi_period_4h).sum() / neg_sum.rolling(self.mfi_period_4h).sum()).replace(0,
                                                                                                                np.nan))))
        # ADX
        out['ADX'] = self._compute_adx(df, self.adx_period_4h)
        out['return_pct'] = df['close'].pct_change() * 100
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    def _compute_indicators_daily(self, df):
        """Compute technical indicators for daily data"""
        out = pd.DataFrame(index=df.index)
        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']
        # Bollinger Bands
        mid = df['close'].rolling(window=self.bb_period_daily).mean()
        std = df['close'].rolling(window=self.bb_period_daily).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + self.bb_stddev_daily * std
        out['BB_lower'] = mid - self.bb_stddev_daily * std
        out['BB_width'] = out['BB_upper'] - out['BB_lower']
        # Historical Volatility
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std()
        # MAs
        for period in self.ma_periods_daily:
            out[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            out[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=self.rsi_period_daily).mean()
        avg_loss = down.rolling(window=self.rsi_period_daily).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f'RSI_{self.rsi_period_daily}'] = 100 - (100 / (1 + rs))
        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast_daily, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow_daily, adjust=False).mean()
        macd_ = ema_fast - ema_slow
        out['MACD'] = macd_
        # ATR
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out[f'ATR_{self.atr_period_daily}'] = tr.rolling(window=self.atr_period_daily).mean()
        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        typical_price_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(typical_price_diff > 0, 0)
        negative_flow = raw_money_flow.where(typical_price_diff < 0, 0)
        pos_sum = positive_flow.rolling(window=self.mfi_period_daily).sum()
        neg_sum = negative_flow.rolling(window=self.mfi_period_daily).sum()
        out[f'MFI_{self.mfi_period_daily}'] = 100 * pos_sum / (pos_sum + neg_sum.replace(0, np.nan))
        # Chaikin Money Flow
        denom = (df['high'] - df['low']).replace(0, np.nan)
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / denom
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(window=self.cmf_period_daily).sum() / df['volume'].rolling(
            window=self.cmf_period_daily).sum().replace(0, np.nan)
        out[f'CMF_{self.cmf_period_daily}'] = cmf
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    def _compute_adx(self, df, period):
        """Compute the Average Directional Index (ADX)"""
        plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
        minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
        tr = (df['high'] - df['low']).combine((df['high'] - df['close'].shift(1)).abs(), np.maximum).combine(
            (df['low'] - df['close'].shift(1)).abs(), np.maximum)
        plus_di = 100 * plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean()
        minus_di = 100 * minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean()
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(window=period).mean()

    def _scale_features(self, df):
        """Scale numeric features using StandardScaler"""
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def _compute_market_regime(self, df):
        """Detect market regime (trending vs ranging)"""
        # Use ADX for trend strength
        adx = df['h4_ADX'].fillna(0)
        # Directional movement
        plus_di = df['h4_SMA_20'].diff().fillna(0)
        minus_di = df['h4_SMA_50'].diff().fillna(0)

        # Define regime: 0=ranging, 1=uptrend, -1=downtrend
        regime = np.zeros(len(df))

        # Strong trend with ADX > 25
        trending_mask = adx > 25
        uptrend_mask = (trending_mask) & (plus_di > 0)
        downtrend_mask = (trending_mask) & (minus_di < 0)

        regime[uptrend_mask] = 1
        regime[downtrend_mask] = -1

        return regime

    def _compute_volume_zones(self, df):
        """Identify high-volume price zones"""
        # Check if the required columns exist and if df is not empty
        if 'close' not in df.columns or 'volume' not in df.columns or df.empty:
            # Return a zero array matching the dataframe length
            return np.zeros(len(df))

        prices = df['close'].values
        volumes = df['volume'].values

        # Check if prices array is empty
        if len(prices) == 0:
            return np.array([])

        # Check if all values are NaN
        if np.all(np.isnan(prices)) or np.all(np.isnan(volumes)):
            return np.zeros(len(df))

        # Filter out NaN values
        valid_mask = ~np.isnan(prices) & ~np.isnan(volumes)
        prices = prices[valid_mask]
        volumes = volumes[valid_mask]

        # Check again if we have data after filtering
        if len(prices) == 0:
            return np.zeros(len(df))

        # Create price bins
        min_price = np.min(prices)
        max_price = np.max(prices)

        # Check if min and max are the same to avoid division by zero
        if min_price == max_price:
            # If all prices are the same, just return a uniformly distributed volume zone
            return np.ones(len(df))

        price_range = max_price - min_price
        bin_size = price_range / 10  # 10 bins

        # Create bins with a small epsilon to ensure max_price is included
        bins = np.arange(min_price, max_price + bin_size * 0.001, bin_size)
        digitized = np.digitize(prices, bins)

        # Sum volume for each bin
        bin_volumes = np.zeros(len(bins))
        for i in range(len(prices)):
            bin_idx = digitized[i] - 1  # adjust for 0-indexing
            if 0 <= bin_idx < len(bin_volumes):
                bin_volumes[bin_idx] += volumes[i]

        # Assign volume zone score for each price point
        vol_zone = np.zeros(len(df))

        # Create a mapping for the valid indices
        valid_indices = np.where(valid_mask)[0]

        for i, valid_idx in enumerate(valid_indices):
            bin_idx = digitized[i] - 1
            if 0 <= bin_idx < len(bin_volumes):
                # Normalize by max volume, with safety check
                if np.max(bin_volumes) > 0:
                    vol_zone[valid_idx] = bin_volumes[bin_idx] / np.max(bin_volumes)
                else:
                    vol_zone[valid_idx] = 0

        return vol_zone
    def _detect_swing_highs(self, df, threshold=0.5):
        """Detect swing highs in price action"""
        highs = df['high'].values
        swing_highs = np.zeros(len(highs))

        # Look for local maxima with surrounding lower prices
        for i in range(5, len(highs) - 5):
            if (highs[i] > highs[i - 1:i]).all() and (highs[i] > highs[i + 1:i + 6]).all():
                # Measure strength by how much higher than neighbors
                left_diff = (highs[i] - np.min(highs[i - 5:i])) / highs[i]
                right_diff = (highs[i] - np.min(highs[i + 1:i + 6])) / highs[i]

                if left_diff > threshold and right_diff > threshold:
                    swing_highs[i] = 1

        return swing_highs

    def _detect_swing_lows(self, df, threshold=0.5):
        """Detect swing lows in price action"""
        lows = df['low'].values
        swing_lows = np.zeros(len(lows))

        # Look for local minima with surrounding higher prices
        for i in range(5, len(lows) - 5):
            if (lows[i] < lows[i - 5:i]).all() and (lows[i] < lows[i + 1:i + 6]).all():
                # Measure strength by how much lower than neighbors
                left_diff = (np.max(lows[i - 5:i]) - lows[i]) / lows[i]
                right_diff = (np.max(lows[i + 1:i + 6]) - lows[i]) / lows[i]

                if left_diff > threshold and right_diff > threshold:
                    swing_lows[i] = 1

        return swing_lows

    def _compute_volatility_regime(self, df):
        """Compute volatility regime"""
        # Use ATR relative to price to determine volatility regime
        atr = df['d1_ATR_14'].values
        close = df['close'].values
        atr_pct = atr / close

        # Calculate percentiles for volatility
        low_vol_threshold = np.nanpercentile(atr_pct, 25)
        high_vol_threshold = np.nanpercentile(atr_pct, 75)

        # Define regime: -1=low vol, 0=normal vol, 1=high vol
        regime = np.zeros(len(df))
        regime[atr_pct < low_vol_threshold] = -1
        regime[atr_pct > high_vol_threshold] = 1

        return regime

    def _compute_mean_reversion(self, df):
        """Compute mean reversion potential"""
        close = df['close'].values
        ma20 = df['h4_SMA_20'].values
        ma50 = df['h4_SMA_50'].values

        # Calculate z-score of price deviation from moving averages
        deviation = (close - (ma20 + ma50) / 2)
        rolling_std = pd.Series(deviation).rolling(window=20).std().values
        z_score = np.zeros_like(deviation)
        mask = rolling_std > 0
        z_score[mask] = deviation[mask] / rolling_std[mask]

        return z_score

    def _compute_trend_strength(self, df):
        """Compute trend strength indicator"""
        adx = df['h4_ADX'].fillna(0).values
        sma_ratio = (df['h4_SMA_20'] / df['h4_SMA_50']).fillna(1).values

        # Combine ADX and SMA ratio for trend strength
        trend_strength = np.zeros(len(df))

        for i in range(len(df)):
            if adx[i] > 30:  # Strong trend
                # Positive when SMA20 > SMA50 (uptrend)
                # Negative when SMA20 < SMA50 (downtrend)
                direction = 1 if sma_ratio[i] > 1 else -1
                strength = min(adx[i] / 100, 1)  # Normalize to -1 to 1
                trend_strength[i] = direction * strength
            else:
                # Weak or no trend
                trend_strength[i] = 0

        return trend_strength

    def _compute_funding_rate_features(self, df, funding_df):
        """
        Compute features from funding rates for futures trading profitability

        Funding rates are crucial for BTC futures trading:
        - Positive funding: Longs pay shorts (bearish indicator)
        - Negative funding: Shorts pay longs (bullish indicator)
        - Extreme funding rates often precede price reversals
        """
        # Create empty DataFrame with price data index
        features = pd.DataFrame(index=df.index)

        if funding_df.empty or 'fundingRate' not in funding_df.columns:
            self.logger.warning("No valid funding rate data available")
            return features

        # Align funding rate data with price data
        aligned_funding = self._align_timeframes(funding_df, df.index)

        if 'fundingRate' not in aligned_funding.columns:
            self.logger.warning("Missing fundingRate column after alignment")
            return features

        try:
            # Raw funding rate
            features['funding_rate'] = aligned_funding['fundingRate']

            # Funding rate momentum (changes)
            features['funding_rate_change'] = features['funding_rate'].diff()
            features['funding_rate_change_3'] = features['funding_rate'].diff(3)  # 3-period change

            # Cumulative funding rate over periods (shows sustained pressure)
            # 3 funding periods = 24 hours (8hr * 3)
            features['funding_cumulative_24h'] = features['funding_rate'].rolling(3).sum()

            # 9 funding periods = 72 hours (8hr * 9) = 3 days
            features['funding_cumulative_3d'] = features['funding_rate'].rolling(9).sum()

            # Z-score of funding rate (identifies extremes)
            # 30 funding periods = 10 days
            mean_funding = features['funding_rate'].rolling(30).mean()
            std_funding = features['funding_rate'].rolling(30).std().replace(0, 0.0001)  # Avoid div/0
            features['funding_zscore'] = (features['funding_rate'] - mean_funding) / std_funding

            # Funding regime identification
            # -1: Consistently negative (bullish)
            # 0: Neutral
            # 1: Consistently positive (bearish)
            features['funding_regime'] = 0
            features.loc[features['funding_cumulative_3d'] < -0.001, 'funding_regime'] = -1
            features.loc[features['funding_cumulative_3d'] > 0.001, 'funding_regime'] = 1

            # Funding rate extreme signal for mean reversion strategies
            # Extremely positive funding rates often precede downward price movements
            # Extremely negative funding rates often precede upward price movements
            features['funding_extreme_signal'] = 0
            features.loc[features['funding_zscore'] > 2.0, 'funding_extreme_signal'] = -1  # Bearish signal
            features.loc[features['funding_zscore'] < -2.0, 'funding_extreme_signal'] = 1  # Bullish signal

            # Funding rate divergence from price (powerful signal)
            # Price rising but funding getting more negative = bullish
            # Price falling but funding getting more positive = bearish
            price_change_5d = df['close'].pct_change(15)  # 15 periods = ~5 days
            funding_direction = np.sign(features['funding_rate'])

            features['funding_divergence'] = 0
            features.loc[(price_change_5d > 0.05) & (funding_direction < 0), 'funding_divergence'] = 1  # Bullish
            features.loc[(price_change_5d < -0.05) & (funding_direction > 0), 'funding_divergence'] = -1  # Bearish

            # Funding rate volatility (indicates market uncertainty)
            features['funding_volatility'] = features['funding_rate'].rolling(9).std()

            # Clean up NaN values
            features = features.fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"Error computing funding rate features: {e}")
            return pd.DataFrame(index=df.index)  # Return empty frame on error

    def _compute_open_interest_features(self, df, oi_df):
        """
        Compute features from open interest data for improved market insights

        Open interest (OI) shows the total outstanding contracts:
        - Rising OI with rising price = strong uptrend
        - Rising OI with falling price = strong downtrend
        - Falling OI = trend exhaustion
        - OI vs. volume reveals market participation dynamics
        """
        features = pd.DataFrame(index=df.index)

        if oi_df.empty or 'sumOpenInterest' not in oi_df.columns:
            self.logger.warning("No valid open interest data available")
            return features

        # Align OI data with price data
        aligned_oi = self._align_timeframes(oi_df, df.index)

        if 'sumOpenInterest' not in aligned_oi.columns:
            self.logger.warning("Missing sumOpenInterest column after alignment")
            return features

        try:
            # Raw open interest
            features['open_interest'] = aligned_oi['sumOpenInterest']

            # Open interest change rates
            features['oi_change_1p'] = features['open_interest'].pct_change()
            features['oi_change_4h'] = features['open_interest'].pct_change(8)  # 8 periods = ~4 hours
            features['oi_change_1d'] = features['open_interest'].pct_change(24)  # 24 periods = ~12 hours

            # Open interest momentum
            features['oi_momentum'] = features['oi_change_1d'].rolling(3).mean()

            # Open interest to volume ratio (participation ratio)
            # Higher values indicate more contract holding vs trading (longer-term sentiment)
            # Lower values indicate more active trading vs holding (shorter-term activity)
            features['oi_volume_ratio'] = features['open_interest'] / df['volume'].replace(0, np.nan)
            features['oi_volume_ratio'] = features['oi_volume_ratio'].fillna(0)

            # OI combined with price direction = sentiment
            # Rising OI + Rising price = strong bullish sentiment
            # Rising OI + Falling price = strong bearish sentiment
            price_direction = np.sign(df['close'].pct_change(8))  # 8 periods = ~4 hours
            oi_direction = np.sign(features['oi_change_4h'])

            features['oi_price_sentiment'] = price_direction * oi_direction

            # OI-based market strength indicator
            oi_strength = np.abs(features['oi_change_1d'])
            features['oi_strength'] = oi_strength * price_direction

            # Clean up NaN values
            features = features.fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"Error computing open interest features: {e}")
            return pd.DataFrame(index=df.index)  # Return empty frame on error
