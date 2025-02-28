import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.um_futures import UMFutures
from binance.error import ClientError
from config import Config

###############################################################################
# DATA FETCHING CLASS
###############################################################################
class BitcoinData:
    def __init__(self, csv_30m='btc_data_30m.csv', csv_4h='btc_data_4h.csv',
                 csv_daily='btc_data_daily.csv', csv_oi='btc_open_interest.csv',
                 csv_funding='btc_funding_rates.csv',
                 base_url="https://fapi.binance.com",
                 timeout=30):
        """
        Initialize with Binance Futures API client and file paths
        """
        # Initialize UMFutures client for USDT-margined futures
        self.client = UMFutures(
            key="XW2qoCu1zlZdA8FEW98y7Md55ZtJ7fpaV1un6QkZErMeHIY7VXYAY5J6FonVTUdy",
            secret="ivuwpI0yYRPRvSrex0IHGLcF4QP6jWTcUMCsWJ0DSQ3retcwOSTcmm9yzB1PFaP2",
            base_url=base_url,
            timeout=timeout
        )

        # CSV file paths
        self.csv_30m = csv_30m
        self.csv_4h = csv_4h
        self.csv_daily = csv_daily
        self.csv_oi = csv_oi
        self.csv_funding = csv_funding

        # Setup logger
        self.logger = logging.getLogger("BitcoinData")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)


    def fetch_30m_data(self, live=False) -> pd.DataFrame:
        """Fetch 30-minute data from file or Binance API"""
        if os.path.exists(self.csv_30m) and not live:
            self.logger.info(f"Loading 30m data from {self.csv_30m}")
            return self._read_csv_with_numeric(self.csv_30m)

        self.logger.info("Fetching 30m data from Binance API")
        try:
            # Calculate start time in milliseconds
            start_time = None
            if hasattr(Config, 'LOOKBACK_30M_CANDLES'):
                start_time = int((datetime.now().timestamp() - (Config.LOOKBACK_30M_CANDLES * 30 * 60)) * 1000)

            # Binance kline data - 30 minute intervals
            klines = self.client.klines(
                symbol="BTCUSDT",
                interval="30m",
                limit=1000,
                startTime=start_time
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Process data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

            # Calculate turnover (quote volume) for compatibility with existing code
            df['turnover'] = df['quote_asset_volume']

            # Select only needed columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'turnover']]

            # Save to CSV
            df.to_csv(self.csv_30m, index=True)
            self.logger.info(f"Fetched {len(df)} 30m candles and saved to {self.csv_30m}")

            return df

        except ClientError as e:
            self.logger.error(f"Binance API error fetching 30m data: {e}")
            # If API fails but we have a CSV file, use it as fallback
            if os.path.exists(self.csv_30m):
                self.logger.info(f"Using existing {self.csv_30m} as fallback")
                return self._read_csv_with_numeric(self.csv_30m)
            raise

    def fetch_open_interest(self, live=False) -> pd.DataFrame:
        """Fetch open interest data from file or Binance API"""
        if os.path.exists(self.csv_oi) and not live:
            self.logger.info(f"Loading open interest data from {self.csv_oi}")
            return pd.read_csv(self.csv_oi, index_col='timestamp', parse_dates=True)

        self.logger.info("Fetching open interest data from Binance API")
        try:
            data = []
            limit = 500  # Maximum allowed by Binance API
            end_time = int(datetime.now().timestamp() * 1000) if live else None

            # We need to make multiple API calls to get enough history
            for _ in range(int(np.ceil(Config.LOOKBACK_30M_CANDLES / limit))):
                params = {
                    "symbol": "BTCUSDT",
                    "period": "30m",  # Match our candle timeframe
                    "limit": limit
                }

                if end_time:
                    params["endTime"] = end_time

                oi_data = self.client.open_interest_hist(**params)

                if not oi_data:
                    break

                data.extend(oi_data)

                if len(oi_data) < limit:
                    break

                # Use the timestamp of the last record minus 1ms as the end time for next batch
                end_time = int(oi_data[-1]['timestamp']) - 1

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Check if df is empty or if expected columns don't exist
            if df.empty:
                self.logger.warning("Received empty data from Binance API for open interest")
                # Create empty DataFrame with expected structure
                empty_df = pd.DataFrame(columns=['timestamp', 'sumOpenInterest'])
                empty_df['timestamp'] = pd.to_datetime(empty_df['timestamp'])
                empty_df.set_index('timestamp', inplace=True)
                return empty_df

            # Check and handle column structure
            # First, inspect what columns we actually have
            self.logger.info(f"Open interest data columns: {df.columns.tolist()}")

            # Convert timestamp to datetime regardless of how it's formatted
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

            # Convert numeric columns, checking first if they exist
            numeric_columns = []

            # Check for openInterest column
            if 'openInterest' in df.columns:
                df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').astype(np.float32)
                numeric_columns.append('openInterest')

            # Check for sumOpenInterest column
            if 'sumOpenInterest' in df.columns:
                df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce').astype(np.float32)
                numeric_columns.append('sumOpenInterest')

            # If we have neither of the expected columns but have other columns, try to adapt
            if not numeric_columns and len(df.columns) > 0:
                # Log what columns we do have
                self.logger.warning(f"Expected columns not found. Available columns: {df.columns.tolist()}")

                # Try to find columns that might contain open interest data
                for col in df.columns:
                    if any(term in col.lower() for term in ['interest', 'oi', 'open']):
                        df['sumOpenInterest'] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                        self.logger.info(f"Using column '{col}' as sumOpenInterest")
                        numeric_columns.append('sumOpenInterest')
                        break

                # If we still don't have any usable columns, create a dummy one
                if not numeric_columns:
                    self.logger.warning("No suitable columns found for open interest. Creating dummy data.")
                    df['sumOpenInterest'] = np.ones(len(df), dtype=np.float32)
                    numeric_columns.append('sumOpenInterest')

            # Save to CSV
            df.to_csv(self.csv_oi, index=True)
            self.logger.info(f"Fetched {len(df)} open interest records and saved to {self.csv_oi}")

            return df

        except ClientError as e:
            self.logger.error(f"Binance API error fetching open interest data: {e}")
            # If API fails but we have a CSV file, use it as fallback
            if os.path.exists(self.csv_oi):
                self.logger.info(f"Using existing {self.csv_oi} as fallback")
                return pd.read_csv(self.csv_oi, index_col='timestamp', parse_dates=True)

            # If all else fails, return an empty DataFrame with expected structure
            self.logger.warning("Creating empty DataFrame for open interest")
            empty_df = pd.DataFrame(columns=['timestamp', 'sumOpenInterest'])
            empty_df['timestamp'] = pd.to_datetime(empty_df['timestamp'])
            empty_df.set_index('timestamp', inplace=True)
            return empty_df

    def fetch_funding_rates(self, live=False) -> pd.DataFrame:
        """Fetch funding rate history from file or Binance API with improved error handling"""
        if os.path.exists(self.csv_funding) and not live:
            self.logger.info(f"Loading funding rate data from {self.csv_funding}")
            return pd.read_csv(self.csv_funding, index_col='timestamp', parse_dates=True)

        self.logger.info("Fetching funding rate data from Binance API")
        try:
            # First, try the direct funding_rate method if it exists
            if hasattr(self.client, 'funding_rate'):
                try:
                    funding_data = self.client.funding_rate(symbol="BTCUSDT", limit=1000)
                    if funding_data:
                        self.logger.info(f"Successfully fetched {len(funding_data)} funding rate records")

                        # Create DataFrame from the results
                        df = pd.DataFrame(funding_data)

                        # Handle timestamp column (typically fundingTime in Binance API)
                        timestamp_col = None
                        for col in ['fundingTime', 'timestamp', 'time']:
                            if col in df.columns:
                                timestamp_col = col
                                break

                        if timestamp_col:
                            df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='ms')
                            df.set_index('timestamp', inplace=True)
                        else:
                            # If no timestamp column found, create one
                            df['timestamp'] = pd.date_range(
                                end=datetime.now(),
                                periods=len(df),
                                freq='8h'  # Using lowercase 'h' to avoid deprecation warning
                            )
                            df.set_index('timestamp', inplace=True)

                        # Ensure fundingRate column exists and is numeric
                        if 'fundingRate' in df.columns:
                            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce').astype(np.float32)
                        else:
                            # If no funding rate column found, create one with default values
                            df['fundingRate'] = 0.0001  # Nominal funding rate

                        # Save to CSV
                        df.to_csv(self.csv_funding, index=True)
                        self.logger.info(f"Fetched {len(df)} funding rate records and saved to {self.csv_funding}")
                        return df
                except Exception as e:
                    self.logger.warning(f"Error using funding_rate method: {e}")

            # If direct method failed, try finding any method with "funding" in the name
            funding_method = None
            available_methods = [method for method in dir(self.client) if not method.startswith('_')]

            self.logger.info(f"Available API methods: {[m for m in available_methods if 'fund' in m.lower()]}")

            for method_name in ['funding_rate_history', 'fundingRate', 'get_funding_rate_history',
                                'futures_funding_rate']:
                if method_name in available_methods:
                    funding_method = method_name
                    self.logger.info(f"Using method '{funding_method}' to fetch funding rate data")
                    break

            # If we found a method, try to use it
            if funding_method:
                try:
                    method_to_call = getattr(self.client, funding_method)
                    funding_data = method_to_call(symbol="BTCUSDT", limit=1000)

                    if funding_data:
                        # Process data similarly to above
                        df = pd.DataFrame(funding_data)

                        # Handle timestamp
                        timestamp_col = None
                        for col in ['fundingTime', 'timestamp', 'time']:
                            if col in df.columns:
                                timestamp_col = col
                                break

                        if timestamp_col:
                            df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='ms')
                            df.set_index('timestamp', inplace=True)
                        else:
                            df['timestamp'] = pd.date_range(
                                end=datetime.now(),
                                periods=len(df),
                                freq='8h'
                            )
                            df.set_index('timestamp', inplace=True)

                        # Ensure fundingRate column exists
                        if 'fundingRate' in df.columns:
                            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce').astype(np.float32)
                        else:
                            df['fundingRate'] = 0.0001

                        df.to_csv(self.csv_funding, index=True)
                        self.logger.info(f"Fetched {len(df)} funding rate records and saved to {self.csv_funding}")
                        return df
                except Exception as e:
                    self.logger.error(f"Error using method {funding_method}: {e}")

            # If all API methods failed, create fallback data
            self.logger.warning("Creating fallback funding rate data")

            # Use 'h' instead of 'H' to avoid deprecation warning
            dates = pd.date_range(
                end=datetime.now(),
                periods=Config.LOOKBACK_30M_CANDLES // 16,
                freq='8h'  # lowercase h
            )

            fallback_data = pd.DataFrame({
                'timestamp': dates,
                'fundingRate': [0.0001] * len(dates)  # Nominal funding rate
            })
            fallback_data.set_index('timestamp', inplace=True)
            fallback_data.to_csv(self.csv_funding, index=True)

            self.logger.info(f"Created fallback funding rate data with {len(fallback_data)} records")
            return fallback_data

        except ClientError as e:
            self.logger.error(f"Binance API error fetching funding rate data: {e}")
            # If API fails but we have a CSV file, use it as fallback
            if os.path.exists(self.csv_funding):
                self.logger.info(f"Using existing {self.csv_funding} as fallback")
                return pd.read_csv(self.csv_funding, index_col='timestamp', parse_dates=True)

            # Create fallback data
            self.logger.warning("Creating fallback funding rate data")
            dates = pd.date_range(
                end=datetime.now(),
                periods=Config.LOOKBACK_30M_CANDLES // 16,
                freq='8h'  # Use lowercase 'h' to avoid deprecation warning
            )

            fallback_data = pd.DataFrame({
                'timestamp': dates,
                'fundingRate': [0.0001] * len(dates)  # Nominal funding rate
            })
            fallback_data.set_index('timestamp', inplace=True)
            fallback_data.to_csv(self.csv_funding, index=True)

            return fallback_data

    def derive_4h_data(self, df_30m) -> pd.DataFrame:
        """Derive 4-hour data from 30-minute data using resampling"""
        # Create a copy to avoid modifying the original DataFrame
        df_30m_copy = df_30m.copy()

        # Ensure the index is datetime and sorted
        if not isinstance(df_30m_copy.index, pd.DatetimeIndex):
            df_30m_copy.index = pd.to_datetime(df_30m_copy.index)
        df_30m_copy = df_30m_copy.sort_index()

        # Resample to 4-hour timeframe - use lowercase 'h' instead of 'H'
        df_4h = df_30m_copy.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum'
        })

        # Drop any NaN rows that might have been created during resampling
        df_4h = df_4h.dropna()

        # Ensure we have the right number of candles
        # If we need specific number of candles:
        if len(df_4h) > Config.LOOKBACK_4H_CANDLES:
            df_4h = df_4h.iloc[-Config.LOOKBACK_4H_CANDLES:]

        # Optionally save to CSV
        df_4h.to_csv(self.csv_4h, index=True)
        self.logger.info(f"Derived {len(df_4h)} 4h candles and saved to {self.csv_4h}")

        return df_4h

    def derive_daily_data(self, df_30m) -> pd.DataFrame:
        """Derive daily data from 30-minute data using resampling"""
        # Create a copy to avoid modifying the original DataFrame
        df_30m_copy = df_30m.copy()

        # Ensure the index is datetime and sorted
        if not isinstance(df_30m_copy.index, pd.DatetimeIndex):
            df_30m_copy.index = pd.to_datetime(df_30m_copy.index)
        df_30m_copy = df_30m_copy.sort_index()

        # Resample to daily timeframe - using 'D' for calendar day
        df_daily = df_30m_copy.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum'
        })

        # Drop any NaN rows that might have been created during resampling
        df_daily = df_daily.dropna()

        # Ensure we have the right number of candles
        if len(df_daily) > Config.LOOKBACK_DAILY_CANDLES:
            df_daily = df_daily.iloc[-Config.LOOKBACK_DAILY_CANDLES:]

        # Optionally save to CSV
        df_daily.to_csv(self.csv_daily, index=True)
        self.logger.info(f"Derived {len(df_daily)} daily candles and saved to {self.csv_daily}")

        return df_daily

    def fetch_liquidation_data(self, live=False) -> pd.DataFrame:
        """Fetch recent liquidation events from Binance API"""
        self.logger.info("Fetching liquidation data from Binance API")
        try:
            # Check if the force_orders method exists
            if not hasattr(self.client, 'force_orders'):
                self.logger.warning("force_orders method not found in Binance client")
                # Check for alternative methods
                available_methods = [method for method in dir(self.client) if not method.startswith('_')]
                liquidation_methods = [m for m in available_methods if 'liquidat' in m.lower() or 'forc' in m.lower()]

                if liquidation_methods:
                    self.logger.info(f"Found alternative liquidation methods: {liquidation_methods}")

                    # Try each method
                    for method_name in liquidation_methods:
                        try:
                            method = getattr(self.client, method_name)
                            # Try with different parameter combinations
                            for params in [
                                {"symbol": "BTCUSDT"},  # Just symbol
                                {"symbol": "BTCUSDT", "limit": 100},  # Symbol and smaller limit
                                {"symbol": "BTCUSDT",
                                 "startTime": int((datetime.now() - timedelta(days=1)).timestamp() * 1000)}
                                # With time constraint
                            ]:
                                try:
                                    self.logger.info(f"Trying {method_name} with params: {params}")
                                    liquidation_data = method(**params)
                                    if liquidation_data:
                                        self.logger.info(f"Successfully fetched liquidation data with {method_name}")
                                        break
                                except Exception as e:
                                    self.logger.warning(f"Failed with params {params}: {e}")

                            if liquidation_data:
                                break
                        except Exception as e:
                            self.logger.warning(f"Method {method_name} failed: {e}")

                # If we still don't have data, return empty DataFrame
                if not liquidation_methods or not liquidation_data:
                    self.logger.warning("No liquidation methods available, returning empty DataFrame")
                    return pd.DataFrame()
            else:
                # Use the original force_orders method but with safer parameters
                try:
                    # Try with a smaller limit value
                    liquidation_data = self.client.force_orders(symbol="BTCUSDT", limit=100)
                except Exception as e:
                    self.logger.warning(f"force_orders failed with limit=100: {e}")
                    # Try without limit parameter
                    try:
                        liquidation_data = self.client.force_orders(symbol="BTCUSDT")
                    except Exception as e:
                        self.logger.warning(f"force_orders failed without limit: {e}")
                        return pd.DataFrame()

            # Convert to DataFrame
            if not liquidation_data:
                return pd.DataFrame()

            df = pd.DataFrame(liquidation_data)

            # Log the columns we received
            self.logger.info(f"Liquidation data columns: {df.columns.tolist()}")

            # Handle diverse column names for timestamp
            timestamp_col = None
            for col in ['time', 'timestamp', 'liquidationTime', 'createTime']:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='ms')
                df.set_index('timestamp', inplace=True)

            # Convert numeric columns if they exist
            numeric_cols = ['price', 'origQty', 'executedQty', 'averagePrice']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

            return df

        except ClientError as e:
            self.logger.error(f"Binance API error fetching liquidation data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def create_fallback_funding_data(self, periods, end_time=None):
        """Helper method to create fallback funding rate data with proper frequency format"""
        if end_time is None:
            end_time = datetime.now()

        # Use lowercase 'h' instead of uppercase 'H'
        dates = pd.date_range(
            end=end_time,
            periods=periods,
            freq='8h'  # lowercase h to avoid deprecation warning
        )

        fallback_data = pd.DataFrame({
            'timestamp': dates,
            'fundingRate': [0.0001] * len(dates)  # Nominal funding rate
        })
        fallback_data.set_index('timestamp', inplace=True)

        return fallback_data

    def _read_csv_with_numeric(self, filepath: str) -> pd.DataFrame:
        """Read CSV file and ensure numeric columns are properly typed"""
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        return df