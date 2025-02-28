import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.src.utils import to_categorical
from gc import collect


###############################################################################
# DATA PREPARATION & SEQUENCE BUILDING
###############################################################################
class CryptoDataPreparer:
    def __init__(self, sequence_length=144, horizon=48, normalize_method='zscore', price_column='close',
                 train_ratio=0.8):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.price_column = price_column
        self.train_ratio = train_ratio
        self.scaler = None

    def prepare_data(self, df):
        """Prepare data for model training, including train/val split with dynamic adjustment for small datasets"""
        df = df.dropna().copy()

        # Store original parameters
        orig_sequence_length = self.sequence_length
        orig_horizon = self.horizon

        # Calculate minimum required data
        min_required = (self.sequence_length + self.horizon) * 1.5  # Need sequence length + horizon, plus buffer

        # If dataset is small, dynamically adjust sequence length and horizon
        if len(df) < min_required:
            # Scale down based on available data
            scaling_factor = max(0.4, len(df) / min_required)  # At least 40% of original

            # Calculate new parameters, ensure minimum values
            new_seq_length = max(4, int(self.sequence_length * scaling_factor))
            new_horizon = max(1, int(self.horizon * scaling_factor))

            print(
                f"Small dataset detected, adjusting parameters: sequence_length={self.sequence_length}->{new_seq_length}, "
                f"horizon={self.horizon}->{new_horizon}")

            # Temporarily adjust parameters
            self.sequence_length = new_seq_length
            self.horizon = new_horizon

        try:
            # Check if we have enough data with possibly adjusted parameters
            if len(df) < (self.sequence_length + self.horizon):
                print(f"Insufficient data: {len(df)} < {self.sequence_length + self.horizon}")
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            df_labeled, labels, fwd_returns_full = self._create_labels(df)
            data_array = df_labeled.values.astype(np.float32)
            X_full, y_full, fwd_returns_full = self._build_sequences(data_array, labels, fwd_returns_full)

            del data_array, labels
            collect()

            if len(X_full) == 0:
                print(f"No sequences could be built. X_full length: 0")
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            # For very small datasets, use a larger portion for training
            actual_train_ratio = self.train_ratio
            if len(X_full) < 20:
                actual_train_ratio = 0.8  # Use 80% for training if very small dataset
                print(f"Very small dataset, using {actual_train_ratio * 100}% for training")

            train_size = max(1, int(actual_train_ratio * len(X_full)))
            X_train, X_val = X_full[:train_size], X_full[train_size:]
            y_train, y_val = y_full[:train_size], y_full[train_size:]
            fwd_returns_val = fwd_returns_full[train_size:]

            entry_indices = list(range(self.sequence_length - 1, len(df_labeled)))
            val_entry_indices = entry_indices[train_size:]

            # Handle edge case where val_entry_indices is empty
            if not val_entry_indices and len(df_labeled) > 0:
                val_entry_indices = [len(df_labeled) - 1]  # Use the last row

            df_val = df_labeled.iloc[val_entry_indices].copy() if val_entry_indices else pd.DataFrame()

            if self.normalize_method and len(X_train) > 0:
                self.scaler = StandardScaler()
                X_train_flat = X_train.reshape(-1, X_train.shape[2])
                self.scaler.fit(X_train_flat)
                X_train = self.scaler.transform(X_train_flat).reshape(X_train.shape)

                # Only transform validation data if it exists
                if len(X_val) > 0:
                    X_val_flat = X_val.reshape(-1, X_val.shape[2])
                    X_val = self.scaler.transform(X_val_flat).reshape(X_val.shape)

                # Add Gaussian noise to training data - reduce for small datasets
                noise_scale = 0.005 if len(X_train) < 20 else 0.01  # Less noise for small datasets
                noise = np.random.normal(0, noise_scale, X_train.shape)
                X_train += noise

            # Oversample extreme classes (0 and 4) - but only if we have enough data
            if len(X_train) >= 10:
                extreme_indices = np.where((y_train[:, 0] == 1) | (y_train[:, 4] == 1))[0]
                if len(extreme_indices) > 0:
                    X_extreme = X_train[extreme_indices]
                    y_extreme = y_train[extreme_indices]
                    X_train = np.concatenate([X_train, X_extreme])
                    y_train = np.concatenate([y_train, y_extreme])

            return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

        finally:
            # Restore original parameters
            self.sequence_length = orig_sequence_length
            self.horizon = orig_horizon

    def prepare_test_data(self, df):
        """Prepare data for testing/inference with dynamic adjustment for small datasets"""
        df = df.dropna().copy()

        # Store original parameters
        orig_sequence_length = self.sequence_length
        orig_horizon = self.horizon

        # If dataset is small, dynamically adjust sequence length and horizon
        if len(df) < (self.sequence_length + self.horizon) * 1.5:
            # Scale down based on available data, preserving ratio
            max_available = len(df)
            scaling_factor = max(0.4, max_available / ((self.sequence_length + self.horizon) * 1.5))

            new_seq_length = max(4, int(self.sequence_length * scaling_factor))
            new_horizon = max(1, int(self.horizon * scaling_factor))

            print(
                f"Small test dataset, adjusting parameters: sequence_length={self.sequence_length}->{new_seq_length}, "
                f"horizon={self.horizon}->{new_horizon}")

            # Temporarily adjust parameters
            self.sequence_length = new_seq_length
            self.horizon = new_horizon

        try:
            # Check if we have enough data with adjusted parameters
            if len(df) < (self.sequence_length + self.horizon):
                print(f"Insufficient test data: {len(df)} < {self.sequence_length + self.horizon}")
                return np.array([]), np.array([]), df, np.array([])

            df_labeled, labels, fwd_returns = self._create_labels(df)
            data_array = df_labeled.values.astype(np.float32)
            X_test, y_test, fwd_returns_test = self._build_sequences(data_array, labels, fwd_returns)

            del data_array, labels
            collect()

            if self.scaler and len(X_test) > 0:
                shape_0, shape_1, shape_2 = X_test.shape
                X_test_flat = X_test.reshape(-1, shape_2)
                X_test = self.scaler.transform(X_test_flat).reshape(shape_0, shape_1, shape_2)

            return X_test, y_test, df_labeled, fwd_returns_test

        finally:
            # Restore original parameters
            self.sequence_length = orig_sequence_length
            self.horizon = orig_horizon

    def _create_labels(self, df):
        """Create classification labels based on forward returns relative to ATR with better handling of small datasets"""
        # Ensure we have enough data
        if len(df) <= self.horizon:
            print(f"Warning: Insufficient data for label creation. Need more than {self.horizon} rows, have {len(df)}.")
            return df.iloc[:-1].copy() if len(df) > 1 else df.copy(), np.zeros(1, dtype=int), np.zeros(1,
                                                                                                       dtype=np.float32)

        # Get the price column
        price = df[self.price_column]

        # Calculate forward returns with explicit handling for NaN values
        future_prices = price.shift(-self.horizon)
        valid_mask = ~price.isna() & ~future_prices.isna()

        # Create a copy of the dataframe up to the point where we have forward data
        # Ensure we don't create an empty dataframe
        end_idx = max(1, len(df) - self.horizon)
        df_copy = df.iloc[:end_idx].copy()

        # Initialize arrays with the correct size
        fwd_return = np.zeros(len(df_copy), dtype=np.float32)
        labels = np.zeros(len(df_copy), dtype=int)

        # Calculate forward returns only for valid entries
        valid_indices = np.where(valid_mask.iloc[:end_idx])[0]
        if len(valid_indices) > 0:
            prices_valid = price.iloc[valid_indices].values
            future_prices_valid = future_prices.iloc[valid_indices].values
            fwd_return[valid_indices] = (future_prices_valid / prices_valid - 1)

        # Add ATR calculation with fallback
        if 'd1_ATR_14' in df.columns:
            atr_pct = (df['d1_ATR_14'] / df[self.price_column]).iloc[:end_idx]
        else:
            # Calculate ATR on the fly as fallback
            high_low = df['high'] - df['low']

            # Adjust lookback period for ATR calculation if dataset is small
            atr_period = min(14, max(3, len(df) // 4))  # Use at least 3 periods, at most 14

            high_close_prev = (df['high'] - df['close'].shift(1)).abs()
            low_close_prev = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            atr_pct = (atr / df['close']).iloc[:end_idx]

        # Fill NaN values in ATR with a reasonable default
        atr_pct = atr_pct.fillna(0.01)  # 1% is a reasonable default

        # Create labels based on ATR
        for i in valid_indices:
            if fwd_return[i] < -2 * atr_pct.iloc[i]:
                labels[i] = 0
            elif fwd_return[i] < -0.5 * atr_pct.iloc[i]:
                labels[i] = 1
            elif fwd_return[i] < 0.5 * atr_pct.iloc[i]:
                labels[i] = 2
            elif fwd_return[i] < 2 * atr_pct.iloc[i]:
                labels[i] = 3
            else:
                labels[i] = 4

        return df_copy, labels, fwd_return

    def _build_sequences(self, data_array, labels_array, fwd_returns_array):
        """Build sequences of fixed length for input to RNN/CNN models with better handling of edge cases"""
        num_samples = len(data_array) - self.sequence_length + 1
        if num_samples <= 0:
            print(
                f"Cannot build sequences: data_array length ({len(data_array)}) < sequence_length ({self.sequence_length})")
            return np.array([]), np.array([]), np.array([])

        X = np.zeros((num_samples, self.sequence_length, data_array.shape[1]), dtype=np.float32)
        y = np.zeros((num_samples, 5), dtype=np.float32)  # For classification; adjust for regression
        fwd_returns = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            X[i] = data_array[i:i + self.sequence_length]
            y[i] = to_categorical(labels_array[i + self.sequence_length - 1], num_classes=5)  # Classification
            # For regression: y[i] = fwd_returns_array[i + self.sequence_length - 1]
            fwd_returns[i] = fwd_returns_array[i + self.sequence_length - 1]

        return X, y, fwd_returns