import logging
from gc import collect

import numpy as np
import pandas as pd
from keras.src.utils import to_categorical


class CryptoDataPreparer:
    """
    Prepares cryptocurrency market data for model training and testing.

    Main methods:
    - prepare_data(df): Prepares data for training, returns (X_train, y_train, X_val, y_val, df_val, fwd_returns_val).
    - prepare_test_data(df): Prepares data for testing/inference.
    """

    def __init__(
            self,
            sequence_length=48,
            horizon=16,
            normalize_method='zscore',
            price_column='close',
            train_ratio=0.7
    ):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.price_column = price_column
        self.train_ratio = train_ratio
        self.scaler = None
        self.feature_names = None
        self.test_sequence_length = sequence_length
        self.feature_importance = None

        self.logger = logging.getLogger("CryptoDataPreparer")

    def prepare_data(self, df):
        """
        Prepare data for model training:
        1. Handle NaNs, infs, and constant/low-var columns.
        2. Adjust sequence/horizon for small or medium datasets.
        3. Create labels and forward returns.
        4. Build sequences for training/validation splits.
        5. Scale features and optionally oversample extremes.
        """
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        critical_columns = ['open', 'high', 'low', 'close', 'volume']

        # Log NaN counts
        nan_counts = df.isna().sum()
        nan_info = nan_counts[nan_counts > 0]
        if len(nan_info) > 0:
            print(f"NaN counts before cleaning:\n{nan_info}")

        # Impute NaNs in non-critical columns, drop rows with NaNs in critical columns
        for col in df.columns:
            if col not in critical_columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        original_len = len(df)
        df.dropna(subset=critical_columns, inplace=True)
        if len(df) < original_len:
            print(f"Dropped {original_len - len(df)} rows with NaNs in critical columns.")

        # Save original parameters
        orig_seq, orig_hor = self.sequence_length, self.horizon

        # Adjust sequence/horizon for small/medium datasets
        if len(df) < 500:
            self.sequence_length = max(8, min(24, int(self.sequence_length * 0.5)))
            self.horizon = max(4, min(8, int(self.horizon * 0.5)))
            print(f"Small dataset. Using sequence_length={self.sequence_length}, horizon={self.horizon}.")
        elif len(df) < 2000:
            self.sequence_length = max(16, min(32, int(self.sequence_length * 0.75)))
            self.horizon = max(8, min(12, int(self.horizon * 0.75)))
            print(f"Medium dataset. Using sequence_length={self.sequence_length}, horizon={self.horizon}.")

        self.test_sequence_length = self.sequence_length

        try:
            if len(df) < (self.sequence_length + self.horizon):
                print(f"Insufficient data: {len(df)} < {self.sequence_length + self.horizon}")
                return (np.array([]),) * 6

            # Drop constant columns
            constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
            if constant_cols:
                print(f"Warning: dropping {len(constant_cols)} constant columns: {constant_cols}")
                df.drop(columns=constant_cols, inplace=True)

            # Warn if highly correlated pairs exist (no dropping here)
            if len(df.columns) > 20:
                try:
                    corr_matrix = df.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr_pairs = [(upper.index[i], upper.columns[j])
                                       for i, j in zip(*np.where(upper > 0.97))]
                    if high_corr_pairs:
                        print(f"Warning: Found {len(high_corr_pairs)} highly correlated feature pairs (>0.97).")
                except Exception as e:
                    print(f"Error calculating correlations: {e}")

            # Drop nearly-constant (very low variance) columns (except critical ones)
            variances = df.var()
            low_var_cols = [c for c in variances.index if variances[c] < 1e-6 and c not in critical_columns]
            if low_var_cols:
                print(f"Warning: dropping {len(low_var_cols)} nearly-constant columns.")
                df.drop(columns=low_var_cols, inplace=True)

            # Create labels, forward returns
            df_labeled, labels, fwd_returns_full = self._create_labels(df)

            # Cap extreme outliers
            for col in df_labeled.select_dtypes(include=np.number).columns:
                Q1, Q3 = df_labeled[col].quantile(0.25), df_labeled[col].quantile(0.75)
                IQR = Q3 - Q1
                low, high = Q1 - 5 * IQR, Q3 + 5 * IQR
                extreme_count = df_labeled[(df_labeled[col] < low) | (df_labeled[col] > high)].shape[0]
                if extreme_count > 0:
                    print(f"Warning: {extreme_count} extreme outliers in '{col}'. Capping values.")
                    df_labeled[col] = df_labeled[col].clip(low, high)

            # Build sequences
            data_array = df_labeled.values.astype(np.float32)
            X_full, y_full, fwd_returns_full = self._build_sequences(data_array, labels, fwd_returns_full)
            del data_array, labels
            collect()

            if len(X_full) == 0:
                print("No sequences built; returning empty arrays.")
                return (np.array([]),) * 6

            # Adjust train ratio for small/medium sets
            actual_train_ratio = self.train_ratio
            if len(X_full) < 200:
                actual_train_ratio = min(0.9, self.train_ratio + 0.1)
            elif len(X_full) < 500:
                actual_train_ratio = min(0.85, self.train_ratio + 0.05)

            train_size = max(1, int(actual_train_ratio * len(X_full)))
            X_train, X_val = X_full[:train_size], X_full[train_size:]
            y_train, y_val = y_full[:train_size], y_full[train_size:]
            fwd_returns_val = fwd_returns_full[train_size:]

            # Indexing for df_val
            entry_indices = list(range(self.sequence_length - 1, len(df_labeled)))
            val_entry_indices = entry_indices[train_size:]
            if not val_entry_indices and len(df_labeled) > 0:
                val_entry_indices = [len(df_labeled) - 1]
            df_val = df_labeled.iloc[val_entry_indices].copy() if val_entry_indices else pd.DataFrame()

            # Feature scaling if requested
            if self.normalize_method and len(X_train) > 0:
                X_train_flat = X_train.reshape(-1, X_train.shape[2])
                variances = np.var(X_train_flat, axis=0)
                zero_var_cols = np.where(variances < 1e-10)[0]
                if len(zero_var_cols) > 0:
                    print(f"Warning: {len(zero_var_cols)} columns with near-zero variance in training.")
                    for c in zero_var_cols:
                        X_train_flat[:, c] = 0.001

                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
                X_train_scaled = self.scaler.fit_transform(X_train_flat)
                self.feature_names = df_labeled.columns.tolist()
                if np.isnan(X_train_scaled).any():
                    print("Warning: NaNs after scaling training data; replacing with 0.")
                    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
                X_train = X_train_scaled.reshape(X_train.shape)

                if len(X_val) > 0:
                    X_val_flat = X_val.reshape(-1, X_val.shape[2])
                    X_val_scaled = self.scaler.transform(X_val_flat)
                    if np.isnan(X_val_scaled).any():
                        print("Warning: NaNs after scaling val data; replacing with 0.")
                        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
                    X_val = X_val_scaled.reshape(X_val.shape)

                # Add small noise to training (data augmentation)
                noise_scale = 0.001 if len(X_train) < 20 else 0.003
                noise = np.random.normal(0, noise_scale, X_train.shape)
                X_train += noise
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                if len(X_val) > 0:
                    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

            # Simple oversampling of extreme classes (if memory allows)
            import psutil
            mem_percent = psutil.virtual_memory().percent
            if len(X_train) >= 50 and mem_percent < 60:
                extreme_indices = np.where((y_train[:, 0] == 1) | (y_train[:, 4] == 1))[0]
                if len(extreme_indices) > 0:
                    max_oversample = min(len(extreme_indices), int(len(X_train) * 0.2))
                    oversample_indices = np.random.choice(extreme_indices, max_oversample, replace=False)
                    X_extreme = X_train[oversample_indices]
                    y_extreme = y_train[oversample_indices]
                    X_train = np.concatenate([X_train, X_extreme])
                    y_train = np.concatenate([y_train, y_extreme])

            print(f"Data prepared. X_train {X_train.shape}, X_val {X_val.shape}")
            print(f"Train label distribution: {np.sum(y_train, axis=0)}")
            print(f"Val label distribution: {np.sum(y_val, axis=0)}")

            zero_train_seqs = np.where(~X_train.any(axis=2))[0]
            if len(zero_train_seqs) > 0:
                print(f"Warning: {len(zero_train_seqs)} all-zero sequences in training.")

            if len(X_val) > 0:
                zero_val_seqs = np.where(~X_val.any(axis=2))[0]
                if len(zero_val_seqs) > 0:
                    print(f"Warning: {len(zero_val_seqs)} all-zero sequences in validation.")

            return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

        finally:
            self.sequence_length = orig_seq
            self.horizon = orig_hor

    def _create_labels(self, df):
        """
        Create classification labels based on forward returns.
        Uses either percentile-based thresholds or ATR-based thresholds,
        depending on dataset size.
        """
        if len(df) <= self.horizon:
            print(f"Warning: Not enough data for label creation. Need > {self.horizon}, have {len(df)}.")
            return df.copy(), np.zeros(1, dtype=int), np.zeros(1, dtype=np.float32)

        price = df[self.price_column]
        future_prices = price.shift(-self.horizon)
        valid_mask = ~price.isna() & ~future_prices.isna()

        end_idx = max(1, len(df) - self.horizon)
        df_copy = df.iloc[:end_idx].copy()

        fwd_return = np.zeros(len(df_copy), dtype=np.float32)
        labels = np.zeros(len(df_copy), dtype=int)
        valid_indices = np.where(valid_mask.iloc[:end_idx])[0]

        if len(valid_indices) > 0:
            p_now = price.iloc[valid_indices].values
            p_future = future_prices.iloc[valid_indices].values
            fwd_return[valid_indices] = (p_future / p_now - 1)

        # Use existing ATR column or compute it
        if 'd1_ATR_14' in df.columns:
            atr_pct = (df['d1_ATR_14'] / df[self.price_column]).iloc[:end_idx].fillna(0.01)
        else:
            high_low = df['high'] - df['low']
            atr_period = min(14, max(3, len(df) // 4))
            high_close_prev = (df['high'] - df['close'].shift(1)).abs()
            low_close_prev = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            atr_pct = (atr / df['close']).iloc[:end_idx].fillna(0.01)

        # Apply different labeling strategies based on dataset size
        if len(valid_indices) > 100:
            percentiles = np.percentile(fwd_return[valid_indices], [20, 40, 60, 80])
            for i in valid_indices:
                if fwd_return[i] < percentiles[0]:
                    labels[i] = 0
                elif fwd_return[i] < percentiles[1]:
                    labels[i] = 1
                elif fwd_return[i] < percentiles[2]:
                    labels[i] = 2
                elif fwd_return[i] < percentiles[3]:
                    labels[i] = 3
                else:
                    labels[i] = 4
        else:
            if len(valid_indices) < 50:
                for i in valid_indices:
                    if fwd_return[i] < -1.0 * atr_pct.iloc[i]:
                        labels[i] = 0
                    elif fwd_return[i] < 1.0 * atr_pct.iloc[i]:
                        labels[i] = 2
                    else:
                        labels[i] = 4
            else:
                for i in valid_indices:
                    if fwd_return[i] < -1.8 * atr_pct.iloc[i]:
                        labels[i] = 0
                    elif fwd_return[i] < -0.4 * atr_pct.iloc[i]:
                        labels[i] = 1
                    elif fwd_return[i] < 0.4 * atr_pct.iloc[i]:
                        labels[i] = 2
                    elif fwd_return[i] < 1.8 * atr_pct.iloc[i]:
                        labels[i] = 3
                    else:
                        labels[i] = 4

        label_counts = np.bincount(labels, minlength=5)
        total_labels = label_counts.sum()
        if total_labels > 0:
            perc = [100.0 * c / total_labels for c in label_counts]
            print(f"Label distribution: 0:{perc[0]:.1f}%, 1:{perc[1]:.1f}%, 2:{perc[2]:.1f}%, "
                  f"3:{perc[3]:.1f}%, 4:{perc[4]:.1f}%")

        return df_copy, labels, fwd_return

    def _build_sequences(self, data_array, labels_array, fwd_returns_array):
        """
        Builds sequences from arrays of features, labels, and forward returns.
        Returns (X, y, fwd_returns).
        """
        num_samples = len(data_array) - self.sequence_length + 1
        if num_samples <= 0:
            print("Cannot build sequences: not enough rows vs. sequence_length.")
            return np.array([]), np.array([]), np.array([])

        X = np.zeros((num_samples, self.sequence_length, data_array.shape[1]), dtype=np.float32)
        y = np.zeros((num_samples, 5), dtype=np.float32)
        fwd_r = np.zeros(num_samples, dtype=np.float32)
        batch_size = min(1000, num_samples)

        try:
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                for i in range(start_idx, end_idx):
                    X[i] = data_array[i: i + self.sequence_length]
                    y[i] = to_categorical(labels_array[i + self.sequence_length - 1], num_classes=5)
                    fwd_r[i] = fwd_returns_array[i + self.sequence_length - 1]

                if (end_idx - start_idx) >= 500:
                    collect()
        except Exception as e:
            print(f"Error building sequences: {e}")

        zero_cols = np.where(~X.any(axis=(0, 1)))[0]
        if len(zero_cols) > 0:
            print(f"Warning: {len(zero_cols)} all-zero feature columns in sequences.")
            for c in zero_cols:
                X[:, :, c] = np.random.normal(0, 0.001, size=(X.shape[0], X.shape[1]))

        zero_ratio = (X == 0).mean(axis=(1, 2))
        high_zero_seqs = np.where(zero_ratio > 0.8)[0]
        if len(high_zero_seqs) > 0:
            print(f"Warning: {len(high_zero_seqs)} sequences with >80% zeros.")

        if np.isnan(X).any():
            print("Warning: NaNs in sequence data; replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0)
        if np.isinf(X).any():
            print("Warning: Inf in sequence data; replacing with zeros.")
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)

        return X, y, fwd_r

    def prepare_test_data(self, df):
        """
        Prepare data for testing/inference:
        1. Clean infs/NaNs similarly to training.
        2. Use the same sequence_length as training (test_sequence_length).
        3. Align columns with training's feature set if available.
        4. Build sequences and optionally scale using the existing scaler.
        """
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        critical_columns = ['open', 'high', 'low', 'close', 'volume']

        for col in df.columns:
            if col not in critical_columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        original_len = len(df)
        df.dropna(subset=critical_columns, inplace=True)
        if len(df) < original_len:
            print(f"Dropped {original_len - len(df)} rows with NaNs in critical columns from test data.")

        orig_seq, orig_hor = self.sequence_length, self.horizon
        self.sequence_length = self.test_sequence_length

        try:
            if len(df) < (self.sequence_length + self.horizon):
                print(f"Insufficient test data: {len(df)} < {self.sequence_length + self.horizon}")
                return np.array([]), np.array([]), df, np.array([])

            # Drop constant columns
            constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
            if constant_cols:
                print(f"Warning: dropping {len(constant_cols)} constant columns in test data.")
                df.drop(columns=constant_cols, inplace=True)

            # Align columns with training if feature_names is set
            if self.feature_names is not None:
                missing_cols = set(self.feature_names) - set(df.columns)
                extra_cols = set(df.columns) - set(self.feature_names)

                if missing_cols:
                    self.logger.warning(f"Test data missing {len(missing_cols)} columns: {missing_cols}")
                    for col in missing_cols:
                        df[col] = 0
                if extra_cols:
                    self.logger.warning(f"Test data has {len(extra_cols)} extra columns: {extra_cols}")
                    df.drop(columns=extra_cols, inplace=True)

                df = df[self.feature_names]
                self.logger.info(f"Test data aligned to {len(self.feature_names)} training features.")

            df_labeled, labels, fwd_returns = self._create_labels(df)

            try:
                X_test, y_test, fwd_returns_test = self._build_sequences(
                    df_labeled.values.astype(np.float32),
                    labels,
                    fwd_returns
                )
            except Exception as e:
                print(f"Error building test sequences: {e}")
                return np.array([]), np.array([]), df_labeled, np.array([])

            del labels
            collect()

            if self.scaler and len(X_test) > 0:
                s0, s1, s2 = X_test.shape
                X_test_flat = X_test.reshape(-1, s2)

                if s2 != self.scaler.n_features_in_:
                    print(f"Feature mismatch: scaler expects {self.scaler.n_features_in_}, got {s2}.")
                    if s2 > self.scaler.n_features_in_:
                        print("Truncating test features to match training dimension.")
                        X_test_flat = X_test_flat[:, :self.scaler.n_features_in_]
                    else:
                        print("Padding test features to match training dimension.")
                        pad_width = self.scaler.n_features_in_ - s2
                        X_test_flat = np.hstack([X_test_flat, np.zeros((len(X_test_flat), pad_width))])

                # Scale in chunks
                batch_size = 5000
                scaled_chunks = []
                for i in range(0, len(X_test_flat), batch_size):
                    chunk = X_test_flat[i:i + batch_size]
                    chunk_scaled = self.scaler.transform(chunk)
                    if np.isnan(chunk_scaled).any():
                        print(f"Warning: NaNs after scaling test chunk {i}-{i + batch_size}.")
                        chunk_scaled = np.nan_to_num(chunk_scaled, nan=0.0)
                    scaled_chunks.append(chunk_scaled)
                    del chunk
                    collect()

                X_test_scaled = np.vstack(scaled_chunks)
                X_test = X_test_scaled.reshape(s0, s1, self.scaler.n_features_in_)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            zero_test_seqs = np.where(~X_test.any(axis=2))[0]
            if len(zero_test_seqs) > 0:
                print(f"Warning: {len(zero_test_seqs)} all-zero sequences in test data.")

            zero_cols = np.where(~X_test.any(axis=(0, 1)))[0]
            if len(zero_cols) > 0:
                print(f"Warning: {len(zero_cols)} all-zero feature columns in test sequences.")
                for c in zero_cols:
                    X_test[:, :, c] = np.random.normal(0, 0.001, size=(X_test.shape[0], X_test.shape[1]))

            print(f"Test data prepared: X_test shape={X_test.shape}.")
            print(f"Test label distribution: {np.sum(y_test, axis=0)}.")

            return X_test, y_test, df_labeled, fwd_returns_test

        finally:
            self.sequence_length = orig_seq
            self.horizon = orig_hor
