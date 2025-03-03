import gc
import logging
import os

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from memory_utils import (
    memory_watchdog,
    log_memory_usage, optimize_memory_for_dataframe
)


###############################################################################
# ENHANCED FEATURE ENGINEERING (REFACTORED)
###############################################################################
class EnhancedCryptoFeatureEngineer:
    def __init__(self, feature_scaling=False, n_jobs=-1):
        """
        Provides methods to generate and manage features for short-term
        Bitcoin trading across multiple timeframes (30m, 4h, daily),
        optionally applying feature selection and scaling.
        """
        self.feature_scaling = feature_scaling
        self.n_jobs = n_jobs

        # Timeframe-specific parameters
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

        self.ma_periods_4h = [20, 50, 100, 200]
        self.rsi_period_4h = 14
        self.macd_fast_4h = 12
        self.macd_slow_4h = 26
        self.macd_signal_4h = 9
        self.mfi_period_4h = 14
        self.adx_period_4h = 14

        self.cmf_period_30m = 20
        self.obv_ma_period_30m = 10
        self.mfi_period_30m = 14
        self.force_ema_span_30m = 2
        self.vwap_period_30m = 20

        # Enhanced parameters
        self.regime_window = 20
        self.volume_zones_lookback = 50
        self.swing_threshold = 0.5

        # Logger setup
        self.logger = logging.getLogger("FeatureEngineer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Track current features & importance
        self.current_features = None
        self.feature_importance = None

    def process_data_3way(self, df_30m, df_4h, df_daily, oi_df=None, funding_df=None):
        """
        Main method to combine features from 30m, 4h, and daily data, plus
        optional open-interest and funding-rate data. Returns a single DataFrame
        of aligned and engineered features.
        """
        log_memory_usage(component="feature_engineering_start")
        memory_watchdog(component="feature_engineering")

        # 1) Optimize input dataframes for memory
        self.logger.info("Optimizing input DataFrames for memory efficiency")
        for df in [df_30m, df_4h, df_daily]:
            optimize_memory_for_dataframe(df, convert_floats=True, convert_ints=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Ensure datetime index without timezone
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        memory_watchdog(component="feature_engineering")

        # 2) Compute 30m indicators
        self.logger.info("Computing 30m indicators")
        feat_30m = self._compute_indicators_30m(df_30m).add_prefix('m30_')
        feat_30m[['open', 'high', 'low', 'close', 'volume']] = df_30m[['open', 'high', 'low', 'close', 'volume']]

        feat_30m = self._fill_nans(feat_30m, timeframe_name="30m")
        del df_30m
        gc.collect()
        memory_watchdog(component="feature_engineering")

        # 3) Compute 4h indicators
        self.logger.info("Computing 4h indicators")
        feat_4h_raw = self._compute_indicators_4h(df_4h).add_prefix('h4_')
        feat_4h_raw = self._fill_nans(feat_4h_raw, timeframe_name="4h")
        feat_4h_ff = self._align_timeframes(feat_4h_raw, feat_30m.index)
        feat_4h_ff = self._fill_nans(feat_4h_ff, timeframe_name="4h_aligned")

        del df_4h, feat_4h_raw
        gc.collect()
        memory_watchdog(component="feature_engineering")

        # 4) Compute daily indicators
        self.logger.info("Computing daily indicators")
        feat_daily_raw = self._compute_indicators_daily(df_daily).add_prefix('d1_')
        feat_daily_raw = self._fill_nans(feat_daily_raw, timeframe_name="daily")
        feat_daily_ff = self._align_timeframes(feat_daily_raw, feat_30m.index)
        feat_daily_ff = self._fill_nans(feat_daily_ff, timeframe_name="daily_aligned")

        del df_daily, feat_daily_raw
        gc.collect()
        memory_watchdog(component="feature_engineering")

        # 5) Combine timeframes
        self.logger.info("Combining features from all timeframes")
        combined = feat_30m.copy()
        for col in feat_4h_ff.columns:
            combined[col] = feat_4h_ff[col]
        for col in feat_daily_ff.columns:
            combined[col] = feat_daily_ff[col]

        del feat_30m, feat_4h_ff, feat_daily_ff
        gc.collect()
        memory_watchdog(component="feature_engineering")

        # 6) Clean up critical data
        combined = self._fill_nans(combined, timeframe_name="combined_all", critical_cols=['open','high','low','close','volume'])
        combined.dropna(subset=['open','high','low','close','volume'], inplace=True)
        if combined.empty:
            self.logger.warning("No data left after dropping NaNs in critical columns.")
            return pd.DataFrame({
                'open': [], 'high': [], 'low': [], 'close': [], 'volume': [],
                'market_regime': [], 'volatility_regime': [], 'trend_strength': [],
                'swing_high': [], 'swing_low': [], 'hist_vol_20': []
            })

        # 7) Replace constant columns with tiny noise
        constant_cols = [c for c in combined.columns if combined[c].nunique() <= 1]
        if constant_cols:
            self.logger.warning(f"Replacing noise in {len(constant_cols)} constant columns: {constant_cols}")
            for col in constant_cols:
                combined[col] += np.random.normal(0, 0.001, size=len(combined))

        # Ensure historical volatility
        if 'hist_vol_20' not in combined.columns:
            try:
                self.logger.info("Computing hist_vol_20 fallback")
                combined['hist_vol_20'] = combined['close'].pct_change().rolling(20).std()
            except Exception as e:
                self.logger.warning(f"Error computing hist_vol_20 fallback: {e}")
                combined['hist_vol_20'] = 0.01

        memory_watchdog(component="feature_engineering")

        # 8) Compute enhanced features in separate batches to manage memory
        # 8A) Market regime
        self.logger.info("Computing market_regime")
        try:
            combined['market_regime'] = self._compute_market_regime(combined)
        except Exception as e:
            self.logger.warning(f"Error computing market regime: {e}")
            combined['market_regime'] = 0

        memory_watchdog(component="feature_engineering")

        # 8B) Funding rate features (if available)
        if funding_df is not None and not funding_df.empty:
            try:
                self.logger.info("Computing funding rate features")
                funding_features = self._compute_funding_rate_features(combined, funding_df)
                for col in funding_features.columns:
                    combined[col] = funding_features[col]
            except Exception as e:
                self.logger.warning(f"Error computing funding rate features: {e}")

        memory_watchdog(component="feature_engineering")

        # 8C) Open interest features (if available)
        if oi_df is not None and not oi_df.empty:
            try:
                self.logger.info("Computing open interest features")
                oi_features = self._compute_open_interest_features(combined, oi_df)
                for col in oi_features.columns:
                    combined[col] = oi_features[col]
            except Exception as e:
                self.logger.warning(f"Error computing open interest features: {e}")

        memory_watchdog(component="feature_engineering")

        # 8D) Volume zones
        self.logger.info("Computing volume zones")
        try:
            combined['volume_zone'] = self._compute_volume_zones(combined)
        except Exception as e:
            self.logger.warning(f"Error computing volume zones: {e}")
            combined['volume_zone'] = 0
        memory_watchdog(component="feature_engineering")

        # 8E) Swing points
        self.logger.info("Detecting swing highs and lows")
        try:
            combined['swing_high'] = self._detect_swing_highs(combined, self.swing_threshold)
            combined['swing_low'] = self._detect_swing_lows(combined, self.swing_threshold)
        except Exception as e:
            self.logger.warning(f"Error detecting swing points: {e}")
            combined['swing_high'] = 0
            combined['swing_low'] = 0
        memory_watchdog(component="feature_engineering")

        # 8F) Final features: volatility regime, mean reversion, trend strength, short-term momentum
        self.logger.info("Computing final advanced features")
        try:
            combined['volatility_regime'] = self._compute_volatility_regime(combined)
            combined['mean_reversion_potential'] = self._compute_mean_reversion(combined)
            combined['trend_strength'] = self._compute_trend_strength(combined)
            combined['momentum_5'] = combined['close'].pct_change(5)
            combined['momentum_10'] = combined['close'].pct_change(10)

            # Volume spike detection
            vol_ma = combined['volume'].rolling(10).mean()
            combined['volume_spike'] = (combined['volume'] > 2 * vol_ma).astype(int)

            # Short-term RSI (5)
            delta = combined['close'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            roll_up, roll_down = up.rolling(5).mean(), down.abs().rolling(5).mean()
            rs = roll_up / roll_down.replace(0, np.nan)
            combined['rsi_5'] = 100.0 - (100.0 / (1.0 + rs))
            combined['rsi_5'].fillna(50, inplace=True)

        except Exception as e:
            self.logger.warning(f"Error computing final features: {e}")
            for fallback_col in ['volatility_regime', 'mean_reversion_potential', 'trend_strength',
                                 'momentum_5', 'momentum_10', 'volume_spike', 'rsi_5']:
                if fallback_col not in combined.columns:
                    combined[fallback_col] = 0 if fallback_col != 'rsi_5' else 50

        memory_watchdog(component="feature_engineering")

        # 9) Clean up final NaNs and Infs
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined.fillna(0, inplace=True)

        # 10) Remove highly correlated features
        combined = self.remove_correlated_features(combined)

        # 11) Adaptive feature selection if memory is high or columns are excessive
        mem = psutil.virtual_memory()
        if mem.percent > 85 or len(combined.columns) > 100:
            self.logger.warning("High memory usage or many features -> using adaptive feature selection")
            try:
                top_features = self.select_top_features(combined, n_top=50)
                self.feature_importance = top_features
            except Exception as e:
                self.logger.warning(f"Error in feature selection: {e}")
                top_features = [c for c in combined.columns if any(x in c for x in
                                ['SMA','EMA','RSI','MACD','ADX','ATR','momentum','volume_spike','rsi_5'])][:50]

            required_cols = ['open','high','low','close','volume','market_regime',
                             'volatility_regime','trend_strength','swing_high','swing_low',
                             'hist_vol_20','momentum_5','volume_spike','rsi_5']
            keep_cols = list(set(required_cols + list(top_features)))
            keep_cols = [c for c in keep_cols if c in combined.columns]
            combined = combined[keep_cols]

            self.logger.info(f"Reduced features from {len(combined.columns)} to {len(keep_cols)}")

        # 12) Ensure certain columns exist
        for col in ['market_regime','volatility_regime','trend_strength','hist_vol_20']:
            if col not in combined.columns:
                combined[col] = 0

        # 13) Scale features if requested
        if self.feature_scaling:
            self.logger.info("Scaling numeric features")
            combined = self._scale_features(combined)

        # 14) Final pass for near-zero variance
        zero_var_cols = combined.var()[lambda x: x < 1e-10].index.tolist()
        if zero_var_cols:
            self.logger.warning(f"Adding noise to {len(zero_var_cols)} near-zero variance columns")
            for col in zero_var_cols:
                combined[col] += np.random.normal(0, 0.001, size=len(combined))

        self.current_features = combined.columns.tolist()
        self.logger.info(f"Feature set finalized with {len(self.current_features)} features")
        log_memory_usage(component="feature_engineering_end")
        return combined

    def remove_correlated_features(self, df, threshold=0.9):
        """
        Remove highly correlated features (|corr| > threshold) to reduce
        multicollinearity. Essential columns remain intact.
        """
        self.logger.info(f"Checking for correlated features > {threshold}...")

        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < 5:
            return df

        try:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if any(upper[c] > threshold)]

            essential_cols = [
                'open','high','low','close','volume','market_regime',
                'volatility_regime','trend_strength','hist_vol_20','momentum_5',
                'volume_spike','rsi_5'
            ]
            to_drop = [c for c in to_drop if c not in essential_cols]

            if to_drop:
                self.logger.info(f"Dropping {len(to_drop)} correlated features: {to_drop}")
                df.drop(columns=to_drop, inplace=True)
            return df

        except Exception as e:
            self.logger.warning(f"Correlation removal error: {e}")
            return df

    def select_top_features(self, df, n_top=50):
        """
        Select top features using a RandomForest-based feature importance
        approach tailored to short-term price moves.
        """
        # Exclude essential columns from selection
        columns_to_exclude = ['open','high','low','close','volume']
        available_features = [c for c in df.columns if c not in columns_to_exclude]

        # Quick checks
        if not available_features:
            self.logger.warning("No columns left to select from after exclusion.")
            return ['close']

        # Define horizon & target for short-term
        horizon = 24
        sample_size = min(5000, len(df))
        if sample_size <= horizon:
            self.logger.warning("Not enough data for reliable feature selection; using default features.")
            return available_features[:n_top]

        try:
            # Binary/ternary target for up/down/flat
            price_change = df['close'].pct_change(horizon).shift(-horizon)
            y = np.where(
                price_change > 0.005, 1,
                np.where(price_change < -0.005, -1, 0)
            )

            valid_mask = ~np.isnan(y)
            y = y[valid_mask]
            X = df.drop(columns=columns_to_exclude, errors='ignore').iloc[valid_mask]

            # Trim to ensure we have no mismatch in row counts from horizon shift
            # We'll remove the last horizon rows that don't have a future label
            X = X.iloc[:-horizon]
            y = y[:-horizon]

            if len(X) < 50:
                self.logger.warning("Too few valid data points for stable feature importance.")
                return available_features[:n_top]

            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=self.n_jobs
            )
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = importances.nlargest(n_top).index.tolist()

            self.logger.info(f"Selected top {len(top_features)} features via RandomForest.")
            return top_features

        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return available_features[:n_top]

    def process_features(self, df_30m, df_4h, df_daily, oi_df=None, funding_df=None,
                         use_chunks=True, chunk_size=1000):
        """
        Single entry point for feature processing, optionally using chunked logic.
        """
        if use_chunks and len(df_30m) > chunk_size:
            self.logger.info(f"Using chunked processing with size {chunk_size}")
            return self.process_data_in_chunks(df_30m, df_4h, df_daily, chunk_size, oi_df, funding_df)
        else:
            self.logger.info("Using direct processing")
            return self.process_data_3way(df_30m, df_4h, df_daily, oi_df, funding_df)

    def process_data_in_chunks(self, df_30m, df_4h, df_daily, chunk_size=1000, oi_df=None, funding_df=None):
        """
        Process data in smaller chunks for memory efficiency, returning a
        combined DataFrame with engineered features.
        """
        self.logger.info(f"Processing data in chunks of size {chunk_size}")
        log_memory_usage(component="chunked_processing_start")
        results = []

        total_chunks = (len(df_30m) + chunk_size - 1) // chunk_size
        self.logger.info(f"Total chunks to process: {total_chunks}")

        for i in range(0, len(df_30m), chunk_size):
            chunk_num = i // chunk_size + 1
            self.logger.info(f"Processing chunk {chunk_num}/{total_chunks}")
            log_memory_usage(component=f"chunk_{chunk_num}_start")

            end_idx = min(i + chunk_size, len(df_30m))
            chunk_30m = df_30m.iloc[i:end_idx]

            start_time, end_time = chunk_30m.index[0], chunk_30m.index[-1]
            chunk_4h = df_4h[df_4h.index <= end_time]
            if not chunk_4h.empty and chunk_4h.index[0] > start_time and len(df_4h) > len(chunk_4h):
                prev_idx_4h = df_4h[df_4h.index < start_time].index
                if len(prev_idx_4h) > 0:
                    chunk_4h = pd.concat([df_4h.loc[[prev_idx_4h[-1]]], chunk_4h])

            chunk_daily = df_daily[df_daily.index <= end_time]
            if not chunk_daily.empty and chunk_daily.index[0] > start_time and len(df_daily) > len(chunk_daily):
                prev_idx_d = df_daily[df_daily.index < start_time].index
                if len(prev_idx_d) > 0:
                    chunk_daily = pd.concat([df_daily.loc[[prev_idx_d[-1]]], chunk_daily])

            chunk_oi = oi_df[(oi_df.index >= start_time) & (oi_df.index <= end_time)] if oi_df is not None else None
            chunk_funding = (funding_df[(funding_df.index >= start_time) & (funding_df.index <= end_time)]
                             if funding_df is not None else None)

            try:
                memory_watchdog(threshold_gb=20, component=f"chunk_{chunk_num}_processing")
                chunk_features = self.process_data_3way(
                    chunk_30m, chunk_4h, chunk_daily, chunk_oi, chunk_funding
                )
                results.append(chunk_features)
                memory_watchdog(threshold_gb=20, component=f"chunk_{chunk_num}_complete")
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_num}: {e}")

            del chunk_30m, chunk_4h, chunk_daily, chunk_oi, chunk_funding, chunk_features
            gc.collect()

        memory_watchdog(threshold_gb=20, component="combining_chunks")

        if not results:
            self.logger.warning("No chunks were processed successfully.")
            return pd.DataFrame()

        log_memory_usage(component="combining_chunks")
        combined = results[0]
        for i in range(1, len(results)):
            self.logger.info(f"Combining chunk {i+1}/{len(results)}")
            combined = pd.concat([combined, results[i]], copy=False)
            results[i] = None
            if i % 3 == 0:
                memory_watchdog(threshold_gb=20, component=f"combined_chunks_{i}")

        del results
        gc.collect()
        log_memory_usage(component="chunked_processing_complete")
        return combined

    ###########################################################################
    # PRIVATE / HELPER METHODS
    ###########################################################################
    def _compute_indicators_30m(self, df):
        """
        Compute technical indicators for 30-minute data, including some additional
        short-term patterns (doji, hammer, shooting star, etc.).
        """
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

        # Historical vol
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std()
        out['hist_vol_5'] = df['close'].pct_change().rolling(5).std()

        # Chaikin Money Flow (CMF)
        multiplier = np.where(
            df['high'] != df['low'],
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']),
            0
        ).astype(np.float32)
        money_flow_volume = multiplier * df['volume']
        out[f'CMF_{self.cmf_period_30m}'] = (money_flow_volume.rolling(self.cmf_period_30m).sum() /
                                             df['volume'].rolling(self.cmf_period_30m).sum()).astype(np.float32)
        out['CMF_5'] = (money_flow_volume.rolling(5).sum() /
                        df['volume'].rolling(5).sum()).astype(np.float32)

        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        out['OBV'] = obv
        out[f'OBV_SMA_{self.obv_ma_period_30m}'] = obv.rolling(self.obv_ma_period_30m).mean()

        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        price_diff = typical_price.diff()
        pos_sum = pd.Series(np.where(price_diff > 0, raw_money_flow, 0), index=df.index).rolling(self.mfi_period_30m).sum()
        neg_sum = pd.Series(np.where(price_diff < 0, raw_money_flow, 0), index=df.index).rolling(self.mfi_period_30m).sum()
        ratio = pos_sum / neg_sum.replace(0, np.nan)
        out[f'MFI_{self.mfi_period_30m}'] = (100 - (100 / (1 + ratio))).astype(np.float32)

        # Short-term MFI (5 periods)
        pos_sum_5 = pd.Series(np.where(price_diff > 0, raw_money_flow, 0), index=df.index).rolling(5).sum()
        neg_sum_5 = pd.Series(np.where(price_diff < 0, raw_money_flow, 0), index=df.index).rolling(5).sum()
        ratio_5 = pos_sum_5 / neg_sum_5.replace(0, np.nan)
        out['MFI_5'] = (100 - (100 / (1 + ratio_5))).astype(np.float32)

        # Force Index (EMA)
        force_index_1 = (df['close'] - df['close'].shift(1)) * df['volume']
        out[f'ForceIndex_EMA{self.force_ema_span_30m}'] = force_index_1.ewm(
            span=self.force_ema_span_30m, adjust=False).mean().astype(np.float32)

        # VWAP
        out[f'VWAP_{self.vwap_period_30m}'] = ((df['close'] * df['volume']).rolling(self.vwap_period_30m).sum() /
                                               df['volume'].rolling(self.vwap_period_30m).sum()).astype(np.float32)

        # Price acceleration
        out['price_accel'] = df['close'].diff().diff()

        # Relative volume
        out['rel_volume'] = df['volume'] / df['volume'].rolling(20).mean()

        # Simple candlestick pattern detections
        doji_threshold = df['high'].rolling(20).mean() * 0.0015
        out['doji'] = (abs(df['close'] - df['open']) < doji_threshold).astype(int)

        body_size = abs(df['close'] - df['open'])
        lower_wick = df[['open','close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open','close']].max(axis=1)
        out['hammer'] = ((lower_wick > 2*body_size) & (upper_wick < 0.5*body_size) & (body_size>0)).astype(int)
        out['shooting_star'] = ((upper_wick > 2*body_size) & (lower_wick < 0.5*body_size) & (body_size>0)).astype(int)
        out['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)

        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    def _compute_indicators_4h(self, df):
        """
        Compute technical indicators for 4-hour data, including Bollinger Bands,
        MAs, RSI, MACD, ADX, MFI, and OBV.
        """
        out = pd.DataFrame(index=df.index)
        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']

        # Bollinger
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + 2 * std
        out['BB_lower'] = mid - 2 * std
        out['BB_width'] = out['BB_upper'] - out['BB_lower']

        # Hist vol
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std()

        # MAs
        for period in self.ma_periods_4h:
            out[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            out[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0).rolling(self.rsi_period_4h).mean()
        down = (-delta.clip(upper=0)).rolling(self.rsi_period_4h).mean()
        rs = up / down.replace(0, np.nan)
        out[f'RSI_{self.rsi_period_4h}'] = 100 - (100/(1+rs))

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
        typical_price = (df['high'] + df['low'] + df['close'])/3
        raw_flow = typical_price * df['volume']
        tp_diff = typical_price.diff()
        pos_flow = raw_flow.where(tp_diff>0, 0)
        neg_flow = raw_flow.where(tp_diff<0, 0)
        out[f'MFI_{self.mfi_period_4h}'] = 100 - (100/(1 + (
            pos_flow.rolling(self.mfi_period_4h).sum() / neg_flow.rolling(self.mfi_period_4h).sum()
        ).replace(0, np.nan)))

        # ADX
        out['ADX'] = self._compute_adx(df, self.adx_period_4h)
        out['return_pct'] = df['close'].pct_change()*100
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    def _compute_indicators_daily(self, df):
        """
        Compute technical indicators for daily data, including Bollinger,
        MAs, RSI, MACD, ATR, MFI, and CMF.
        """
        out = pd.DataFrame(index=df.index)
        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']

        mid = df['close'].rolling(window=self.bb_period_daily).mean()
        std = df['close'].rolling(window=self.bb_period_daily).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + self.bb_stddev_daily * std
        out['BB_lower'] = mid - self.bb_stddev_daily * std
        out['BB_width'] = out['BB_upper'] - out['BB_lower']

        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std()

        for period in self.ma_periods_daily:
            out[f'SMA_{period}'] = df['close'].rolling(period).mean()
            out[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0).rolling(self.rsi_period_daily).mean()
        down = (-delta.clip(upper=0)).rolling(self.rsi_period_daily).mean()
        rs = up / down.replace(0, np.nan)
        out[f'RSI_{self.rsi_period_daily}'] = 100 - (100/(1+rs))

        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast_daily, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow_daily, adjust=False).mean()
        out['MACD'] = ema_fast - ema_slow

        # ATR
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out[f'ATR_{self.atr_period_daily}'] = tr.rolling(self.atr_period_daily).mean()

        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        typical_diff = typical_price.diff()
        pos_flow = raw_money_flow.where(typical_diff>0, 0)
        neg_flow = raw_money_flow.where(typical_diff<0, 0)
        pos_sum = pos_flow.rolling(self.mfi_period_daily).sum()
        neg_sum = neg_flow.rolling(self.mfi_period_daily).sum()
        out[f'MFI_{self.mfi_period_daily}'] = 100 * pos_sum / (pos_sum + neg_sum.replace(0, np.nan))

        # CMF
        denom = (df['high'] - df['low']).replace(0, np.nan)
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / denom
        mf_volume = mf_multiplier * df['volume']
        out[f'CMF_{self.cmf_period_daily}'] = mf_volume.rolling(self.cmf_period_daily).sum() / \
                                              df['volume'].rolling(self.cmf_period_daily).sum().replace(0, np.nan)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    def _compute_adx(self, df, period):
        """
        Compute the Average Directional Index (ADX) for the given DataFrame df
        over the specified period.
        """
        plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
        minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)

        plus_di = 100 * plus_dm.rolling(period).mean() / tr.rolling(period).mean()
        minus_di = 100 * minus_dm.rolling(period).mean() / tr.rolling(period).mean()

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()

    def _scale_features(self, df):
        """
        Scale numeric features using StandardScaler.
        """
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def _compute_market_regime(self, df):
        """
        Detect market regime (trending vs ranging) using ADX and short MAs.
        Values: 1=uptrend, -1=downtrend, 0=none.
        """
        try:
            # If large DF, try parallel chunk processing
            if len(df) > 10000:
                from joblib import Parallel, delayed
                chunk_size = min(5000, len(df)//(self.n_jobs*2) if self.n_jobs>1 else 5000)
                chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._compute_regime_chunk)(chunk) for chunk in chunks
                )
                return np.concatenate(results)
        except:
            pass

        adx = df.get('h4_ADX', pd.Series(0, index=df.index)).fillna(0)
        plus_di = df.get('h4_SMA_20', pd.Series(0, index=df.index)).diff().fillna(0)
        minus_di = df.get('h4_SMA_50', pd.Series(0, index=df.index)).diff().fillna(0)

        regime = np.zeros(len(df))
        strong_trend = adx > 25

        uptrend_mask = strong_trend & (plus_di > 0)
        downtrend_mask = strong_trend & (minus_di < 0)
        regime[uptrend_mask] = 1
        regime[downtrend_mask] = -1
        return regime

    def _compute_regime_chunk(self, chunk):
        """
        Helper for parallel chunk-based regime detection.
        """
        adx = chunk.get('h4_ADX', pd.Series(0, index=chunk.index)).fillna(0)
        plus_di = chunk.get('h4_SMA_20', pd.Series(0, index=chunk.index)).diff().fillna(0)
        minus_di = chunk.get('h4_SMA_50', pd.Series(0, index=chunk.index)).diff().fillna(0)

        regime = np.zeros(len(chunk))
        strong_trend = adx > 25
        uptrend_mask = strong_trend & (plus_di > 0)
        downtrend_mask = strong_trend & (minus_di < 0)
        regime[uptrend_mask] = 1
        regime[downtrend_mask] = -1
        return regime

    def _compute_volume_zones(self, df):
        """
        Identify high-volume price zones by binning close prices and summing volumes.
        Returns a normalized volume-zone score for each row.
        """
        if ('close' not in df.columns) or ('volume' not in df.columns) or df.empty:
            return np.zeros(len(df))

        prices = df['close'].values
        volumes = df['volume'].values
        valid_mask = ~np.isnan(prices) & ~np.isnan(volumes)
        if not np.any(valid_mask):
            return np.zeros(len(df))

        prc_valid = prices[valid_mask]
        vol_valid = volumes[valid_mask]
        if len(prc_valid) == 0:
            return np.zeros(len(df))

        min_p, max_p = np.min(prc_valid), np.max(prc_valid)
        if min_p == max_p:
            return np.ones(len(df))

        price_range = max_p - min_p
        bin_size = price_range / 10
        bins = np.arange(min_p, max_p + bin_size*0.001, bin_size)
        digitized = np.digitize(prc_valid, bins)
        bin_volumes = np.zeros(len(bins))

        for i, bin_idx in enumerate(digitized):
            idx = bin_idx - 1
            if 0 <= idx < len(bin_volumes):
                bin_volumes[idx] += vol_valid[i]

        vol_zone = np.zeros(len(df))
        valid_indices = np.where(valid_mask)[0]
        for i, v_idx in enumerate(valid_indices):
            bidx = digitized[i] - 1
            if 0 <= bidx < len(bin_volumes) and np.max(bin_volumes)>0:
                vol_zone[v_idx] = bin_volumes[bidx] / np.max(bin_volumes)
        return vol_zone

    def _detect_swing_highs(self, df, threshold=0.5):
        """
        Detect local maxima with enough difference from neighbors
        to count as a strong swing high.
        """
        highs = df['high'].values
        swing_highs = np.zeros(len(highs))
        for i in range(5, len(highs)-5):
            if (highs[i] > highs[i-1:i].max()) and (highs[i] > highs[i+1:i+6].max()):
                left_diff = (highs[i] - highs[i-5:i].min()) / highs[i]
                right_diff = (highs[i] - highs[i+1:i+6].min()) / highs[i]
                if left_diff > threshold and right_diff > threshold:
                    swing_highs[i] = 1
        return swing_highs

    def _detect_swing_lows(self, df, threshold=0.5):
        """
        Detect local minima with enough difference from neighbors
        to count as a strong swing low.
        """
        lows = df['low'].values
        swing_lows = np.zeros(len(lows))
        for i in range(5, len(lows)-5):
            if (lows[i] < lows[i-5:i].min()) and (lows[i] < lows[i+1:i+6].min()):
                left_diff = (lows[i-5:i].max() - lows[i]) / lows[i]
                right_diff = (lows[i+1:i+6].max() - lows[i]) / lows[i]
                if left_diff > threshold and right_diff > threshold:
                    swing_lows[i] = 1
        return swing_lows

    def _compute_volatility_regime(self, df):
        """
        Classify volatility regime based on d1_ATR_14 relative to price.
        -1 = low vol, 0 = normal, 1 = high vol.
        """
        atr = df.get('d1_ATR_14', pd.Series(0, index=df.index)).values
        close = df['close'].values
        atr_pct = atr / close
        low_thr = np.nanpercentile(atr_pct, 25)
        high_thr = np.nanpercentile(atr_pct, 75)

        regime = np.zeros(len(df))
        regime[atr_pct < low_thr] = -1
        regime[atr_pct > high_thr] = 1
        return regime

    def _compute_mean_reversion(self, df):
        """
        Rough measure of how far current price is from short MAs,
        returning a 'z-score' style metric.
        """
        close = df['close'].values
        ma20 = df.get('h4_SMA_20', pd.Series(close, index=df.index)).values
        ma50 = df.get('h4_SMA_50', pd.Series(close, index=df.index)).values
        dev = close - (ma20 + ma50)/2
        rolling_std = pd.Series(dev).rolling(20).std().values
        z_score = np.zeros_like(dev)
        mask = rolling_std > 0
        z_score[mask] = dev[mask]/rolling_std[mask]
        return z_score

    def _compute_trend_strength(self, df):
        """
        Combine ADX & short MAs to produce a trend-strength indicator in [-1..1].
        """
        adx = df.get('h4_ADX', pd.Series(0, index=df.index)).fillna(0).values
        sma_ratio = (df.get('h4_SMA_20', pd.Series(1, index=df.index)) /
                     df.get('h4_SMA_50', pd.Series(1, index=df.index))).fillna(1).values

        strength = np.zeros(len(df))
        for i in range(len(df)):
            if adx[i] > 30:
                direction = 1 if sma_ratio[i]>1 else -1
                # Normalize ADX to [0..1]
                adx_val = min(adx[i]/100, 1)
                strength[i] = direction * adx_val
        return strength

    def _compute_funding_rate_features(self, df, funding_df):
        """
        Compute features from funding rates for futures trading (e.g. extremes, momentum, etc.).
        """
        features = pd.DataFrame(index=df.index)
        if funding_df.empty or 'fundingRate' not in funding_df.columns:
            return features

        aligned_funding = self._align_timeframes(funding_df, df.index)
        if 'fundingRate' not in aligned_funding.columns:
            return features

        try:
            features['funding_rate'] = aligned_funding['fundingRate']
            features['funding_rate_change'] = features['funding_rate'].diff()
            features['funding_rate_change_3'] = features['funding_rate'].diff(3)
            features['funding_cumulative_24h'] = features['funding_rate'].rolling(3).sum()
            features['funding_cumulative_3d'] = features['funding_rate'].rolling(9).sum()

            mean_f = features['funding_rate'].rolling(30).mean()
            std_f = features['funding_rate'].rolling(30).std().replace(0, 0.0001)
            features['funding_zscore'] = (features['funding_rate'] - mean_f)/std_f

            features['funding_regime'] = 0
            features.loc[features['funding_cumulative_3d'] < -0.001, 'funding_regime'] = -1
            features.loc[features['funding_cumulative_3d'] > 0.001, 'funding_regime'] = 1

            features['funding_extreme_signal'] = 0
            features.loc[features['funding_zscore']>2.0, 'funding_extreme_signal'] = -1
            features.loc[features['funding_zscore']<-2.0, 'funding_extreme_signal'] = 1

            price_change_5d = df['close'].pct_change(15)
            fund_sign = np.sign(features['funding_rate'])
            features['funding_divergence'] = 0
            features.loc[(price_change_5d>0.05) & (fund_sign<0), 'funding_divergence'] = 1
            features.loc[(price_change_5d<-0.05) & (fund_sign>0), 'funding_divergence'] = -1

            features['funding_volatility'] = features['funding_rate'].rolling(9).std()
            features.fillna(0, inplace=True)
            return features

        except Exception as e:
            self.logger.error(f"Error computing funding features: {e}")
            return pd.DataFrame(index=df.index)

    def _compute_open_interest_features(self, df, oi_df):
        """
        Compute features from open interest data (OI).
        E.g. OI changes, momentum, OI/volume ratio, sentiment alignment, etc.
        """
        features = pd.DataFrame(index=df.index)
        if oi_df.empty or 'sumOpenInterest' not in oi_df.columns:
            return features

        aligned_oi = self._align_timeframes(oi_df, df.index)
        if 'sumOpenInterest' not in aligned_oi.columns:
            return features

        try:
            features['open_interest'] = aligned_oi['sumOpenInterest']
            features['oi_change_1p'] = features['open_interest'].pct_change()
            features['oi_change_4h'] = features['open_interest'].pct_change(8)
            features['oi_change_1d'] = features['open_interest'].pct_change(24)
            features['oi_momentum'] = features['oi_change_1d'].rolling(3).mean()

            vol_nonzero = df['volume'].replace(0, np.nan)
            features['oi_volume_ratio'] = features['open_interest']/vol_nonzero
            features['oi_volume_ratio'].fillna(0, inplace=True)

            price_dir = np.sign(df['close'].pct_change(8))
            oi_dir = np.sign(features['oi_change_4h'])
            features['oi_price_sentiment'] = price_dir * oi_dir

            oi_strength = np.abs(features['oi_change_1d'])
            features['oi_strength'] = oi_strength * price_dir

            features.fillna(0, inplace=True)
            return features

        except Exception as e:
            self.logger.error(f"Error computing OI features: {e}")
            return pd.DataFrame(index=df.index)

    def _align_timeframes(self, higher_tf_data, target_index):
        """
        Align higher timeframe data to a lower timeframe's index by forward-filling
        each candle's values until the next candle. Returns a DataFrame with the same
        index as target_index.
        """
        if higher_tf_data.empty or len(target_index) == 0:
            return pd.DataFrame(index=target_index)

        aligned_data = pd.DataFrame(index=target_index)
        numeric_data = higher_tf_data.select_dtypes(include=np.number).copy()
        numeric_data.ffill().bfill()

        last_checkpoint = 0
        row_count = 0
        idx_vals = numeric_data.index

        for idx, row in numeric_data.iterrows():
            row_count += 1
            if row_count - last_checkpoint >= 100:
                memory_watchdog(threshold_gb=15, force_cleanup=False, component="timeframe_alignment")
                last_checkpoint = row_count

            if idx == idx_vals[-1]:
                mask = (target_index >= idx)
            else:
                next_idx = idx_vals[idx_vals.get_loc(idx) + 1]
                mask = (target_index >= idx) & (target_index < next_idx)

            for col in numeric_data.columns:
                if col not in aligned_data:
                    aligned_data[col] = np.nan
                aligned_data.loc[mask, col] = row[col]

        aligned_data.ffill(inplace=True)
        aligned_data.bfill(inplace=True)
        aligned_data.fillna(0, inplace=True)
        return aligned_data

    ###########################################################################
    # Small helper for repeated NaN filling and logging
    ###########################################################################
    def _fill_nans(self, df, timeframe_name="generic", critical_cols=None):
        """
        Fill NaN values in DataFrame 'df' with forward/backward fill,
        and optionally log or drop if 'critical_cols' is provided.
        """
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            self.logger.warning(f"NaNs in {timeframe_name}: {null_cols}")
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df.fillna(0, inplace=True)

        if critical_cols:
            old_size = len(df)
            df.dropna(subset=critical_cols, inplace=True)
            if len(df) < old_size:
                self.logger.warning(
                    f"Dropped {old_size - len(df)} rows in {timeframe_name} due to missing critical cols."
                )
        return df
