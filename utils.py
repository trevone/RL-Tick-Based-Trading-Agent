# utils.py
import yaml
import os
import pandas as pd
import numpy as np
import traceback
import json
import hashlib
from datetime import datetime, timezone, timedelta
import time
from typing import Union
import re

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    print("CRITICAL ERROR: python-binance library not found. This project now exclusively uses Binance for data. "
          "Please install with 'pip install python-binance' for the scripts to function.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not found. Technical indicators will not be calculated. "
          "Please install with 'pip install TA-Lib' for full functionality.")


# --- NEW: Data Cache Directory ---
# Consistent with check_tick_cache.py
DATA_CACHE_DIR = "./binance_data_cache/"

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}. Using script fallbacks if available.")
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"Error loading YAML configuration from {config_path}: {e}")
        return {}

def _calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: list) -> pd.DataFrame:
    if not TALIB_AVAILABLE:
        print("TA-Lib not available, skipping technical indicator calculation.")
        # Ensure all requested features are present, even if as NaNs
        for feature in price_features_to_add:
            if feature not in df.columns and feature not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[feature] = np.nan
        return df

    if df.empty:
        return df
    df_processed = df.copy()

    # Ensure required columns are present and numeric
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Cannot calculate TAs.")
            df_processed[col] = np.nan # Use NaN first to allow TA-Lib to handle it
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # Ensure numeric

    # Drop rows with NaN in critical columns at the start to prevent TA-Lib errors
    df_processed.dropna(subset=['High', 'Low', 'Close'], inplace=True)
    if df_processed.empty:
        print("WARNING: DataFrame became empty after dropping NaNs for TA calculation.")
        return df_processed

    # Prepare numpy arrays from pandas Series for TA-Lib
    high_np = df_processed['High'].values
    low_np = df_processed['Low'].values
    close_np = df_processed['Close'].values
    open_np = df_processed['Open'].values
    volume_np = df_processed['Volume'].values

    # --- Dynamic TA-Lib Calculations for Indicators and Candlestick Patterns ---
    for feature_name in price_features_to_add:
        # Check if the feature is one of the base OHLCV columns (already handled)
        if feature_name in required_cols_for_ta:
            continue

        try:
            # Moving Averages
            if feature_name.startswith('SMA_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.SMA(close_np, timeperiod=timeperiod)

            elif feature_name.startswith('EMA_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.EMA(close_np, timeperiod=timeperiod)

            # Oscillators
            elif feature_name.startswith('RSI_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.RSI(close_np, timeperiod=timeperiod)

            elif feature_name == 'MACD':
                macd, macdsignal, macdhist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
                df_processed['MACD'] = macd

            elif feature_name == 'ADX':
                df_processed['ADX'] = talib.ADX(high_np, low_np, close_np, timeperiod=14)
            
            elif feature_name == 'STOCH_K':
                stoch_k, stoch_d = talib.STOCH(high_np, low_np, close_np,
                                              fastk_period=5, slowk_period=3, slowd_period=3)
                df_processed['STOCH_K'] = stoch_k
            
            # Volatility Indicators
            elif feature_name == 'ATR':
                df_processed['ATR'] = talib.ATR(high_np, low_np, close_np, timeperiod=14)
            
            elif feature_name == 'BBANDS_Upper':
                upper, middle, lower = talib.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                df_processed['BBANDS_Upper'] = upper

            # Volume Indicators
            elif feature_name == 'AD':
                df_processed['AD'] = talib.AD(high_np, low_np, close_np, volume_np)

            elif feature_name == 'OBV':
                df_processed['OBV'] = talib.OBV(close_np, volume_np)
            
            # Candlestick Patterns
            elif feature_name.startswith('CDL'):
                if hasattr(talib, feature_name):
                    pattern_func = getattr(talib, feature_name)
                    if feature_name == 'CDLMORNINGSTAR':
                        df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np, penetration=0)
                    elif feature_name == 'CDLEVENINGSTAR':
                        df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np, penetration=0)
                    else:
                        df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np)
                else:
                    print(f"Warning: TA-Lib function '{feature_name}' not found for candlestick pattern.")
                    df_processed[feature_name] = np.nan

            else:
                print(f"Warning: Technical indicator or pattern '{feature_name}' requested but no specific calculation logic defined. Assigning NaN.")
                df_processed[feature_name] = np.nan

        except Exception as e:
            print(f"Error calculating TA '{feature_name}': {e}. Assigning NaN.")
            traceback.print_exc()
            df_processed[feature_name] = np.nan

    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True)
    
    for feature in price_features_to_add:
        if feature not in df_processed.columns and feature not in required_cols_for_ta:
            print(f"Warning: Requested feature '{feature}' is still missing after all calculations and fills. Setting to 0.0.")
            df_processed[feature] = 0.0

    return df_processed

def fetch_and_cache_kline_data(
    symbol: str, interval: str, start_date_str: str, end_date_str: str,
    cache_dir: str, 
    price_features_to_add: list = None,
    api_key: str = None, api_secret: str = None, testnet: bool = False,
    cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2
) -> pd.DataFrame:
    """
    Fetches and caches Binance K-line (OHLCV) data, optionally adding technical indicators.
    """
    if not BINANCE_CLIENT_AVAILABLE:
        print("CRITICAL ERROR in fetch_and_cache_kline_data: python-binance library not found.")
        return pd.DataFrame()

    price_features_to_add = price_features_to_add if price_features_to_add is not None else []
    os.makedirs(cache_dir, exist_ok=True)

    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")

    cache_file = get_data_path_for_day(
        date_str=daily_file_date_str, 
        symbol=symbol, 
        data_type="kline", 
        interval=interval, 
        price_features_to_add=price_features_to_add, 
        cache_dir=cache_dir
    )

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: print(f"Loading K-line data from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')
            
            expected_output_columns = set(['Open', 'High', 'Low', 'Close', 'Volume'] + price_features_to_add)
            
            if not all(col in df.columns for col in expected_output_columns):
                print(f"Warning: Cached K-line data missing some requested TA features or base columns. Refetching: {cache_file}")
                os.remove(cache_file)
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error loading K-line data from cache {cache_file}: {e}. Refetching.")
            if os.path.exists(cache_file): os.remove(cache_file)
            return pd.DataFrame()

    if log_level in ["normal", "detailed"]:
        print(f"Fetching K-line data for {symbol}, Interval: {interval}, From: {start_date_str}, To: {end_date_str}")

    client = Client(api_key or os.environ.get('BINANCE_API_KEY'), 
                    api_secret or os.environ.get('BINANCE_API_SECRET'), 
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    try:
        klines_raw = client.get_historical_klines(symbol, interval, start_date_str, end_str=end_date_str)
        if not klines_raw:
            if log_level != "none": print(f"No k-lines returned for {symbol} {start_date_str}-{end_date_str} for interval {interval}.")
            return pd.DataFrame()

        kline_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 
                        'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore']
        df_fetched = pd.DataFrame(klines_raw, columns=kline_cols)
        
        df_fetched['OpenTimeDate'] = pd.to_datetime(df_fetched['OpenTime'], unit='ms', utc=True)
        df_fetched.set_index('OpenTimeDate', inplace=True)

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df_fetched.columns: df_fetched[col] = pd.to_numeric(df_fetched[col], errors='coerce')
        
        df_to_process = df_fetched[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_to_process = df_to_process.astype(float)

        df_with_ta = _calculate_technical_indicators(df_to_process, price_features_to_add)
        
        if not df_with_ta.empty:
            try:
                if cache_file_type == "parquet": df_with_ta.to_parquet(cache_file)
                else: df_with_ta.to_csv(cache_file)
                if log_level in ["normal", "detailed"]: print(f"K-line data with TAs saved to cache: {cache_file}")
            except Exception as e: print(f"Error saving K-line data to cache {cache_file}: {e}")
        
        return df_with_ta

    except BinanceAPIException as bae:
        print(f"Binance API Exception during K-line fetch: Code={bae.code}, Message='{bae.message}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error during K-line fetch/processing: {e}")
        traceback.print_exc(); return pd.DataFrame()


def fetch_continuous_aggregate_trades(
    symbol: str, start_date_str: str, end_date_str: str, 
    cache_dir: str, api_key: str = None, api_secret: str = None, 
    testnet: bool = False, cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2
) -> pd.DataFrame:
    """
    Fetches and caches continuous aggregate trade data for a given symbol and date range.
    Returns the DataFrame if data is fetched, an empty DataFrame if no data, or raises an exception on error.
    """
    if not BINANCE_CLIENT_AVAILABLE:
        print("CRITICAL ERROR in fetch_continuous_aggregate_trades: python-binance library not found.")
        return pd.DataFrame()

    os.makedirs(cache_dir, exist_ok=True)
    
    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")
    cache_file = get_data_path_for_day(daily_file_date_str, symbol, data_type="agg_trades", cache_dir=cache_dir)

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: print(f"Loading aggregate trades from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')
            return df
        except Exception as e:
            print(f"Error loading aggregate trades from cache {cache_file}: {e}. Refetching.")
            if os.path.exists(cache_file): os.remove(cache_file)
            return pd.DataFrame()

    if log_level in ["normal", "detailed"]:
        print(f"Fetching continuous aggregate trades for {symbol} from {start_date_str} to {end_date_str}.")

    client = Client(api_key or os.environ.get('BINANCE_API_KEY'), 
                    api_secret or os.environ.get('BINANCE_API_SECRET'), 
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    all_trades = []
    start_dt = pd.to_datetime(start_date_str, utc=True)
    end_dt = pd.to_datetime(end_date_str, utc=True)
    
    current_start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    while True:
        try:
            if log_level == "detailed":
                print(f"  Fetching agg trades. Current start_ms: {current_start_ms}")

            chunk_end_ms = min(current_start_ms + (60 * 60 * 1000 - 1), end_ms)

            if log_level == "detailed": print(f"    Fetching chunk: Start {datetime.fromtimestamp(current_start_ms/1000, tz=timezone.utc)}, End {datetime.fromtimestamp(chunk_end_ms/1000, tz=timezone.utc)}")

            trades_chunk = client.get_aggregate_trades(
                symbol=symbol, 
                startTime=current_start_ms,
                endTime=chunk_end_ms,
                limit=1000
            )
            
            if trades_chunk:
                all_trades.extend(trades_chunk)
                current_start_ms = trades_chunk[-1]['T'] + 1 
                if log_level == "detailed": print(f"    Fetched {len(trades_chunk)} trades in chunk. New start_ms: {current_start_ms}")
            else:
                if current_start_ms < end_ms:
                     current_start_ms = chunk_end_ms + 1
                     if log_level == "detailed": print(f"    No trades in chunk, advancing window. New start_ms: {current_start_ms}")
                else:
                    break
            
            if current_start_ms >= end_ms:
                break
            
            time.sleep(api_request_delay_seconds)

        except BinanceAPIException as bae:
            print(f"Binance API Exception during aggregate trade fetch: {bae.code} - {bae.message}. Retrying or stopping...")
            time.sleep(api_request_delay_seconds * 5)
            break 
        except Exception as e:
            print(f"Error fetching aggregate trades: {e}")
            traceback.print_exc()
            break

    if not all_trades:
        if log_level != "none": print(f"No aggregate trades returned for {symbol} {start_date_str}-{end_date_str}.")
        return pd.DataFrame()

    df_trades = pd.DataFrame(all_trades)
    df_trades.rename(columns={'T': 'Timestamp', 'p': 'Price', 'q': 'Quantity', 'm': 'IsBuyerMaker', 'a': 'TradeId'}, inplace=True)
    df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'], unit='ms', utc=True)
    df_trades.set_index('Timestamp', inplace=True)
    df_trades[['Price', 'Quantity']] = df_trades[['Price', 'Quantity']].astype(float)
    
    df_trades_final = df_trades[['Price', 'Quantity', 'IsBuyerMaker']].copy()

    if not df_trades_final.empty:
        try:
            if cache_file_type == "parquet": df_trades_final.to_parquet(cache_file)
            else: df_trades_final.to_csv(cache_file)
            if log_level in ["normal", "detailed"]: print(f"Aggregate trades saved to cache: {cache_file}")
            time.sleep(0.1)
        except Exception as e: print(f"Error saving aggregate trades to cache {cache_file}: {e}")
    else:
        if log_level != "none": print(f"DataFrame is empty, not saving to cache: {cache_file}")
            
    return df_trades_final

# --- UPDATED: Helper for file paths for daily data ---
def get_data_path_for_day(date_str: str, symbol: str, data_type: str = "agg_trades", 
                          interval: str = None, price_features_to_add: list = None, 
                          cache_dir: str = DATA_CACHE_DIR) -> str:
    """
    Generates the expected file path for a single day's data (e.g., aggregate trades or klines).
    Includes interval and features in filename for klines.
    The filename format is simplified to:
    - agg_trades: bn_aggtrades_SYMBOL_YYYY-MM-DD.parquet
    - kline: bn_klines_SYMBOL_INTERVAL_YYYY-MM-DD_FEATURES.parquet
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    if data_type == "agg_trades":
        filename_prefix = "bn_aggtrades"
        # Simplified: bn_aggtrades_SYMBOL_YYYY-MM-DD.parquet
        safe_filename = f"{filename_prefix}_{symbol}_{date_str}.parquet"
    elif data_type == "kline":
        if not interval:
            raise ValueError("Interval must be provided for kline data type.")
        
        # Sort features to ensure consistent filename for caching
        sorted_features_str = ""
        if price_features_to_add:
            # Normalize feature names for consistent hashing/filename (e.g., remove non-alphanumeric, lowercase)
            # This logic needs to align with how parse_filename_for_metadata reconstructs them.
            # It's safest to convert to a consistent, simple format for the filename.
            normalized_features_for_filename = []
            for f in sorted(price_features_to_add): # Sort to ensure consistent filename
                normalized_features_for_filename.append(re.sub(r'[^a-zA-Z0-9]', '', f).lower())
            
            sorted_features_str = "_".join(normalized_features_for_filename)
            if sorted_features_str:
                sorted_features_str = f"_{sorted_features_str}" # Add underscore only if features exist

        filename_prefix = "bn_klines"
        # Simplified: bn_klines_SYMBOL_INTERVAL_YYYY-MM-DD_FEATURES.parquet
        safe_filename = f"{filename_prefix}_{symbol}_{interval}_{date_str}{sorted_features_str}.parquet"
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Must be 'agg_trades' or 'kline'.")
    
    return os.path.join(cache_dir, safe_filename)


# --- NEW: Function to fetch and cache data for a single day (used by data_downloader_manager.py) ---
def fetch_and_cache_tick_data(symbol: str, start_date_str: str, end_date_str: str, cache_dir: str = DATA_CACHE_DIR) -> Union[pd.DataFrame, None]:
    """
    Wrapper to fetch and cache aggregate trade data for a specific date range (typically a single day).
    This function is intended to be called by the new data_downloader_manager.py for daily fetching.
    It calls fetch_continuous_aggregate_trades.
    Returns the DataFrame if data was fetched and saved, otherwise None.
    """
    df = fetch_continuous_aggregate_trades(
        symbol=symbol,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        cache_dir=cache_dir,
        api_key=None,
        api_secret=None,
        testnet=False,
        log_level="normal",
        api_request_delay_seconds=0.2
    )
    return df if not df.empty else None

# --- NEW: Function to load data for training, checking for missing days ---
def load_tick_data_for_range(symbol: str, start_date_str: str, end_date_str: str, cache_dir: str = DATA_CACHE_DIR) -> pd.DataFrame:
    """
    Loads tick data (aggregate trades) for a given symbol and date range.
    Checks for and fetches any missing days within the range for training purposes.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    all_data_frames = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_path = get_data_path_for_day(date_str, symbol, data_type="agg_trades", cache_dir=cache_dir)

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Missing or empty data for {symbol} on {date_str}. Attempting to fetch and cache.")
            try:
                day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
                day_end_dt_utc = datetime.combine(current_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(microseconds=1)
                start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                df_fetched_daily = fetch_and_cache_tick_data(symbol, start_datetime_str_for_api, end_datetime_str_for_api, cache_dir=cache_dir)
                
                if df_fetched_daily is None or df_fetched_daily.empty:
                    print(f"No data was fetched for {symbol} on {date_str}. Skipping this day.")
                    current_date += timedelta(days=1)
                    continue
            except Exception as e:
                print(f"Could not fetch missing data for {symbol} on {date_str}: {e}. Skipping this day.")
                current_date += timedelta(days=1)
                continue

        try:
            df = pd.read_parquet(file_path)
            if not df.empty:
                all_data_frames.append(df)
            else:
                print(f"File {file_path} is empty after re-check. Skipping this day.")
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this day.")

        current_date += timedelta(days=1)

    if not all_data_frames:
        print(f"No data found or loaded for {symbol} in the range {start_date_str} to {end_date_str}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames)
    combined_df = combined_df.sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    return combined_df


def load_kline_data_for_range(symbol: str, start_date_str: str, end_date_str: str, interval: str, 
                              price_features: list, cache_dir: str = DATA_CACHE_DIR) -> pd.DataFrame:
    """
    Loads kline data for a given symbol, interval, and date range.
    Checks for and fetches any missing days within the range.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    all_data_frames = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_path = get_data_path_for_day(date_str, symbol, data_type="kline", 
                                          interval=interval, price_features_to_add=price_features, 
                                          cache_dir=cache_dir)

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Missing or empty K-line data for {symbol} on {date_str} (Interval: {interval}). Attempting to fetch and cache.")
            try:
                day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
                day_end_dt_utc = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc)
                start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S")

                df_fetched_daily = fetch_and_cache_kline_data(
                    symbol=symbol,
                    interval=interval,
                    start_date_str=start_datetime_str_for_api,
                    end_date_str=end_datetime_str_for_api,
                    cache_dir=cache_dir,
                    price_features_to_add=price_features
                )
                
                if df_fetched_daily is None or df_fetched_daily.empty:
                    print(f"No K-line data was fetched for {symbol} on {date_str} for interval {interval}. Skipping this day.")
                    current_date += timedelta(days=1)
                    continue
            except Exception as e:
                print(f"Could not fetch missing K-line data for {symbol} on {date_str} (Interval: {interval}): {e}. Skipping this day.")
                current_date += timedelta(days=1)
                continue

        try:
            df = pd.read_parquet(file_path)
            if not df.empty:
                all_data_frames.append(df)
            else:
                print(f"File {file_path} is empty after re-check. Skipping this day.")
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this day.")

        current_date += timedelta(days=1)

    if not all_data_frames:
        print(f"No K-line data found or loaded for {symbol} in the range {start_date_str} to {end_date_str} for interval {interval}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames)
    combined_df = combined_df.sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    return combined_df


def merge_configs(default_config: dict, loaded_config: dict, section_name: str = "") -> dict:
    merged = default_config.copy()
    if loaded_config:
        for key, value in loaded_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value, f"{section_name}.{key if section_name else key}")
            else:
                merged[key] = value
    return merged

def generate_config_hash(config_dict: dict, length: int = 7) -> str:
    config_string = json.dumps(convert_to_native_types(config_dict), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def convert_to_native_types(data):
    if isinstance(data, list): return [convert_to_native_types(item) for item in data]
    if isinstance(data, dict): return {key: convert_to_native_types(value) for key, value in data.items()}
    if isinstance(data, np.integer): return int(data)
    if isinstance(data, np.floating): return float(data)
    if isinstance(data, np.ndarray): return data.tolist() 
    if isinstance(data, (np.bool_, bool)): return bool(data)
    if isinstance(data, pd.Timestamp): return data.isoformat()
    return data

def get_relevant_config_for_hash(
        full_yaml_config: dict,
        train_script_fallback_config: dict, 
        env_script_fallback_config: dict 
    ) -> dict:
    effective_train_cfg = train_script_fallback_config 
    effective_env_cfg = effective_train_cfg.get("environment", env_script_fallback_config)
    effective_bn_cfg = effective_train_cfg.get("binance_settings", {})
    effective_ppo_cfg = train_script_fallback_config.get("ppo_params", {})

    hash_keys_structure = full_yaml_config.get("hash_config_keys", {}) 

    relevant_config_for_hash = {}

    if "environment" in hash_keys_structure and isinstance(hash_keys_structure["environment"], list):
        env_keys_to_hash = hash_keys_structure["environment"]
        relevant_config_for_hash["environment"] = {
            k: effective_env_cfg.get(k) for k in env_keys_to_hash if k in effective_env_cfg
        }

    if "ppo_params" in hash_keys_structure and isinstance(hash_keys_structure["ppo_params"], list):
        ppo_keys_to_hash = hash_keys_structure["ppo_params"]
        relevant_config_for_hash["ppo_params"] = {
            k: effective_ppo_cfg.get(k) for k in ppo_keys_to_hash if k in effective_ppo_cfg
        }
        if "policy_kwargs" in relevant_config_for_hash.get("ppo_params", {}) and \
           isinstance(relevant_config_for_hash["ppo_params"]["policy_kwargs"], str):
            try:
                relevant_config_for_hash["ppo_params"]["policy_kwargs"] = eval(relevant_config_for_hash["ppo_params"]["policy_kwargs"])
            except Exception:
                pass

    if "binance_settings" in hash_keys_structure and isinstance(hash_keys_structure["binance_settings"], list):
        bn_keys_to_hash = hash_keys_structure["binance_settings"]
        relevant_config_for_hash["binance_settings"] = {
            k: effective_bn_cfg.get(k) for k in bn_keys_to_hash if k in effective_bn_cfg
        }
    return {k: v for k, v in relevant_config_for_hash.items() if v}


def resolve_model_path(eval_specific_config: dict, full_yaml_config: dict, train_script_fallback_config: dict, env_script_fallback_config: dict, log_level: str = "normal") -> tuple:
    model_load_path = eval_specific_config.get("model_path") or eval_specific_config.get("model_load_path")
    alt_model_load_path = eval_specific_config.get("alt_model_path")

    if model_load_path and os.path.exists(model_load_path) and model_load_path.endswith(".zip"):
        if log_level in ["normal", "detailed"]: print(f"Using explicit model_path: {model_load_path}")
        return model_load_path, alt_model_load_path
    elif model_load_path: 
         if log_level != "none": print(f"Warning: Explicit model_path '{model_load_path}' not found or invalid. Attempting reconstruction.")

    if log_level in ["normal", "detailed"]: print("Attempting to reconstruct model path from training config hash...")
    
    temp_effective_config_for_hash = merge_configs(train_script_fallback_config, full_yaml_config) 
    relevant_config_for_hash = get_relevant_config_for_hash(full_yaml_config, temp_effective_config_for_hash, env_script_fallback_config)

    train_log_dir_base = temp_effective_config_for_hash.get("run_settings", {}).get("log_dir_base")
    train_base_model_name = temp_effective_config_for_hash.get("run_settings", {}).get("model_name")

    if train_log_dir_base and train_base_model_name:
        if not relevant_config_for_hash and log_level != "none": print("Warning: Cannot auto-find model, relevant config for hash is empty.")
        else:
            config_hash = generate_config_hash(relevant_config_for_hash)
            final_model_name_with_hash = f"{config_hash}_{train_base_model_name}"
            expected_run_dir = os.path.join(train_log_dir_base, final_model_name_with_hash)
            
            if log_level == "detailed": print(f"Expected run directory for model (based on current config for hash): {expected_run_dir}")
            if log_level == "detailed": print(f"  Relevant parts for hash were: {json.dumps(convert_to_native_types(relevant_config_for_hash), indent=2, sort_keys=True)}")


            path_best_model = os.path.join(expected_run_dir, "best_model", "best_model.zip")
            path_final_model = os.path.join(expected_run_dir, "trained_model_final.zip")

            if os.path.exists(path_best_model):
                if log_level in ["normal", "detailed"]: print(f"Found best model: {path_best_model}")
                return path_best_model, path_final_model if os.path.exists(path_final_model) else alt_model_load_path
            elif os.path.exists(path_final_model):
                if log_level in ["normal", "detailed"]: print(f"Found final model: {path_final_model}")
                return path_final_model, alt_model_load_path
            elif log_level != "none": print(f"No standard models found in {expected_run_dir} (check hash and paths).")
    elif log_level != "none": print("Cannot reconstruct: training log_dir_base or model_name missing in current effective config.")
    
    if log_level != "none": print("No valid model path found through explicit path or reconstruction.")
    return None, alt_model_load_path


if __name__ == '__main__':
    print("--- Testing utils.py (New Structure with Tick and K-line Fetchers) ---")

    cfg_test = load_config()
    temp_default_train_config_for_test = {
        "run_settings": {"log_level": "normal", "log_dir_base": "./logs/ppo_trading_test/", "model_name": "test_model"},
        "environment": {
            "kline_price_features": ["Open", "High", "Low", "Close", "Volume", "SMA_10", "RSI_14"]
        },
        "ppo_params": {"total_timesteps": 100},
        "binance_settings": {
            "default_symbol": "BTCUSDT",
            "historical_interval": "1h",
            "historical_cache_dir": "./binance_data_cache/",
            "api_key": os.environ.get("BINANCE_API_KEY"),
            "api_secret": os.environ.get("BINANCE_API_SECRET"),
            "testnet": True,
            "api_request_delay_seconds": 0.1
        },
        "hash_config_keys": {
            "environment": ["kline_price_features"],
            "binance_settings": ["default_symbol", "historical_interval"]
        }
    }

    test_config_merged = merge_configs(temp_default_train_config_for_test, cfg_test)
    test_config_merged["environment"] = merge_configs(test_config_merged["environment"], cfg_test.get("environment", {}))


    current_utils_log_level = test_config_merged.get("run_settings", {}).get("log_level")
    bn_settings_test = test_config_merged.get("binance_settings", {})
    env_settings_test_features = test_config_merged.get("environment", {}).get("kline_price_features", [])

    if BINANCE_CLIENT_AVAILABLE:
        print("\n--- Testing fetch_and_cache_kline_data ---")
        kline_df = fetch_and_cache_kline_data(
            symbol=bn_settings_test.get("default_symbol"),
            interval=bn_settings_test.get("historical_interval"),
            start_date_str=test_config_merged.get("start_date_kline_test", "2024-04-01 00:00:00"),
            end_date_str=test_config_merged.get("end_date_kline_test", "2024-04-01 23:59:59"),
            cache_dir=bn_settings_test.get("historical_cache_dir"),
            price_features_to_add=env_settings_test_features,
            api_key=bn_settings_test.get("api_key"),
            api_secret=bn_settings_test.get("api_secret"),
            testnet=bn_settings_test.get("testnet"),
            log_level=current_utils_log_level,
            api_request_delay_seconds=bn_settings_test.get("api_request_delay_seconds")
        )
        if not kline_df.empty:
            print(f"K-line fetch successful. Shape: {kline_df.shape}, Columns: {kline_df.columns.tolist()}")
            print("K-line Head:\n", kline_df.head())
        else:
            print("K-line fetch FAILED or returned empty DataFrame.")

        print("\n--- Testing fetch_continuous_aggregate_trades (original function) ---")
        tick_start = test_config_merged.get("start_date_tick_test", "2024-04-01 10:00:00")
        tick_end = test_config_merged.get("end_date_tick_test", "2024-04-01 10:05:00")
        
        print(f"Fetching aggregate trades for {bn_settings_test.get('default_symbol')} from {tick_start} to {tick_end}")

        agg_trades_df = fetch_continuous_aggregate_trades(
            symbol=bn_settings_test.get("default_symbol"),
            start_date_str=tick_start,
            end_date_str=tick_end,
            cache_dir=bn_settings_test.get("historical_cache_dir"),
            api_key=bn_settings_test.get("api_key"),
            api_secret=bn_settings_test.get("api_secret"),
            testnet=bn_settings_test.get("testnet"),
            log_level=current_utils_log_level,
            api_request_delay_seconds=bn_settings_test.get("api_request_delay_seconds")
        )
        if not agg_trades_df.empty:
            print(f"Aggregate trades fetch successful. Shape: {agg_trades_df.shape}, Columns: {agg_trades_df.columns.tolist()}")
            print("Agg Trades Head:\n", agg_trades_df.head())
        else:
            print("Aggregate trades fetch FAILED or returned empty DataFrame.")
            print("Note: For aggregate trades, ensure the test period has trading activity on Binance Testnet (if used) or Mainnet.)")

        print("\n--- Testing NEW fetch_and_cache_tick_data (daily wrapper) ---")
        daily_test_date_agg = "2024-04-03"
        print(f"Attempting to fetch and cache aggregate trades data for {daily_test_date_agg} using the daily wrapper...")
        try:
            test_day_dt_agg = datetime.strptime(daily_test_date_agg, '%Y-%m-%d').date()
            test_day_start_dt_utc_agg = datetime.combine(test_day_dt_agg, datetime.min.time(), tzinfo=timezone.utc)
            test_day_end_dt_utc_agg = datetime.combine(test_day_dt_agg + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(microseconds=1)
            test_start_datetime_str_for_api_agg = test_day_start_dt_utc_agg.strftime("%Y-%m-%d %H:%M:%S")
            test_end_datetime_str_for_api_agg = test_day_end_dt_utc_agg.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            fetched_daily_df_agg = fetch_and_cache_tick_data(
                symbol=bn_settings_test.get("default_symbol"),
                start_date_str=test_start_datetime_str_for_api_agg,
                end_date_str=test_end_datetime_str_for_api_agg,
                cache_dir=DATA_CACHE_DIR
            )
            if fetched_daily_df_agg is not None and not fetched_daily_df_agg.empty:
                print(f"Daily aggregate trades fetch and cache for {daily_test_date_agg} complete. Check {get_data_path_for_day(daily_test_date_agg, bn_settings_test.get('default_symbol'))}")
            else:
                print(f"Daily aggregate trades fetch and cache for {daily_test_date_agg} completed but no data was fetched or DataFrame is empty.")
        except Exception as e:
            print(f"Daily aggregate trades fetch and cache for {daily_test_date_agg} FAILED: {e}")

        print("\n--- Testing NEW load_tick_data_for_range ---")
        range_start_agg = "2024-04-01"
        range_end_agg = "2024-04-03"
        print(f"Attempting to load aggregate trades data for range {range_start_agg} to {range_end_agg}...")
        range_df_agg = load_tick_data_for_range(
            symbol=bn_settings_test.get("default_symbol"),
            start_date_str=range_start_agg,
            end_date_str=range_end_agg,
            cache_dir=DATA_CACHE_DIR
        )
        if not range_df_agg.empty:
            print(f"Aggregate trades range load successful. Shape: {range_df_agg.shape}")
            print("Aggregate Trades Range Data Head:\n", range_df_agg.head())
            print("Aggregate Trades Range Data Tail:\n", range_df_agg.tail())
        else:
            print("Aggregate trades range load FAILED or returned empty DataFrame.")

        print("\n--- Testing NEW load_kline_data_for_range ---")
        kline_range_start = "2024-04-01"
        kline_range_end = "2024-04-02"
        kline_test_interval = "1h"
        kline_test_features = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "RSI_14"]
        print(f"Attempting to load K-line data for range {kline_range_start} to {kline_range_end} (Interval: {kline_test_interval}, Features: {kline_test_features})...")
        range_df_kline = load_kline_data_for_range(
            symbol=bn_settings_test.get("default_symbol"),
            start_date_str=kline_range_start,
            end_date_str=kline_range_end,
            interval=kline_test_interval,
            price_features=kline_test_features,
            cache_dir=DATA_CACHE_DIR
        )
        if not range_df_kline.empty:
            print(f"K-line range load successful. Shape: {range_df_kline.shape}")
            print("K-line Range Data Head:\n", range_df_kline.head())
            print("K-line Range Data Tail:\n", range_df_kline.tail())
        else:
            print("K-line range load FAILED or returned empty DataFrame.")


    else:
        print("Skipped Binance data fetching tests as python-binance library is not available.")
    
    print("\n--- Utils.py testing finished ---")