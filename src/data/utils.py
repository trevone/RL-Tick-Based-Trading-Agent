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
from typing import Union, List, Dict
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
DATA_CACHE_DIR = "data_cache/" # Updated path

def _load_single_yaml_config(config_path: str) -> Dict:
    """Loads a single YAML configuration file."""
    if not os.path.exists(config_path):
        # This is often expected for default config files that might not exist for all combinations
        # print(f"Warning: Configuration file not found at {config_path}.") # Removed warning for common use
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"Error loading YAML configuration from {config_path}: {e}")
        return {}

def load_config(main_config_path: str = "config.yaml",
                default_config_paths: List[str] = None) -> Dict:
    """
    Loads and merges configuration from multiple YAML files.
    Default config paths are loaded first (lowest priority), then main_config_path (highest priority).
    """
    if default_config_paths is None:
        default_config_paths = []

    # Start with an empty config
    merged_config = {}

    # Load and merge default configurations
    for path in default_config_paths:
        default_cfg = _load_single_yaml_config(path)
        merged_config = merge_configs(merged_config, default_cfg)

    # Load and merge the main configuration, overriding defaults
    main_cfg = _load_single_yaml_config(main_config_path)
    merged_config = merge_configs(merged_config, main_cfg)

    return merged_config


def _calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: list) -> pd.DataFrame:
    df_processed = df.copy()

    # Ensure required columns are present and numeric
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Cannot calculate TAs.")
            # If critical column is missing, fill with NaN and return to prevent further errors
            df_processed[col] = np.nan
            df_processed.fillna(0, inplace=True) # Fill all to avoid further NaNs
            return df_processed
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # Ensure numeric

    # Drop rows with NaN in critical columns at the start to prevent TA-Lib errors
    df_processed.dropna(subset=['High', 'Low', 'Close'], inplace=True)
    if df_processed.empty:
        print("WARNING: DataFrame became empty after dropping NaNs for TA calculation.")
        return df_processed

    if not TALIB_AVAILABLE:
        print("TA-Lib not available, skipping technical indicator calculation.")
        # Only return the base OHLCV features if TA-Lib is not available
        final_df = df_processed[required_cols_for_ta].copy()
        return final_df.bfill().ffill().fillna(0) # Fill NaNs for base features


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
                    df_processed[feature_name] = np.nan # Still add with NaN if func not found
            else:
                print(f"Warning: Technical indicator or pattern '{feature_name}' requested but no specific calculation logic defined. Assigning NaN.")
                df_processed[feature_name] = np.nan # Still add with NaN if logic not found

        except Exception as e:
            print(f"Error calculating TA '{feature_name}': {e}. Assigning NaN.")
            traceback.print_exc()
            df_processed[feature_name] = np.nan

    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True)
    
    # Final check for requested features, especially for cases where TA-Lib func not found.
    # This loop is redundant with the TALIB_AVAILABLE=False block above if features are only TAs
    # For safety, keep to ensure ALL requested features are in final output.
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
            
            # Re-process if TA-Lib is now available and features missing, or if column count is off
            # This check is better for ensuring consistency: if a required TA feature is missing, refetch.
            # Base OHLCV features are always expected from API, TAs might be added.
            expected_output_columns = set([col for col in price_features_to_add if col in ['Open', 'High', 'Low', 'Close', 'Volume']] + 
                                          [col for col in price_features_to_add if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and TALIB_AVAILABLE])

            if not expected_output_columns.issubset(df.columns):
                print(f"Warning: Cached K-line data missing some requested features (e.g., TAs) or columns mismatch. Refetching: {cache_file}")
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
        api_key=None, # These should be set via env vars or client init outside this func
        api_secret=None, # These should be set via env vars or client init outside this func
        testnet=False, # This should be set via config
        log_level="normal", # This should be set via config
        api_request_delay_seconds=0.2 # This should be set via config
    )
    return df if not df.empty else None

# --- NEW: Function to load data for training, checking for missing days ---
def load_tick_data_for_range(symbol: str, start_date_str: str, end_date_str: str, cache_dir: str = DATA_CACHE_DIR,
                             binance_settings: Dict = None) -> pd.DataFrame: # Added binance_settings
    """
    Loads tick data (aggregate trades) for a given symbol and date range.
    Checks for and fetches any missing days within the range for training purposes.
    """
    if binance_settings is None:
        binance_settings = {} # Use an empty dict if not provided

    # The format string for strptime needs to match the input string exactly.
    # The start_date_str and end_date_str from binance_settings are 'YYYY-MM-DD HH:MM:SS'.
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date() # Corrected format for end_date_str

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

                df_fetched_daily = fetch_continuous_aggregate_trades(
                    symbol=symbol,
                    start_date_str=start_datetime_str_for_api,
                    end_date_str=end_datetime_str_for_api,
                    cache_dir=cache_dir,
                    api_key=binance_settings.get("api_key"),
                    api_secret=binance_settings.get("api_secret"),
                    testnet=binance_settings.get("testnet", False),
                    log_level=binance_settings.get("log_level", "normal"),
                    api_request_delay_seconds=binance_settings.get("api_request_delay_seconds", 0.2)
                )
                
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
                              price_features: list, cache_dir: str = DATA_CACHE_DIR,
                              binance_settings: Dict = None) -> pd.DataFrame: # Added binance_settings
    """
    Loads kline data for a given symbol, interval, and date range.
    Checks for and fetches any missing days within the range.
    """
    if binance_settings is None:
        binance_settings = {} # Use an empty dict if not provided

    # The format string for strptime needs to match the input string exactly.
    # The start_date_str and end_date_str from binance_settings are 'YYYY-MM-DD HH:MM:SS'.
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date() # Corrected format for end_date_str

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
                    price_features_to_add=price_features,
                    api_key=binance_settings.get("api_key"),
                    api_secret=binance_settings.get("api_secret"),
                    testnet=binance_settings.get("testnet", False),
                    log_level=binance_settings.get("log_level", "normal"),
                    api_request_delay_seconds=binance_settings.get("api_request_delay_seconds", 0.2)
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


def merge_configs(default_config: Dict, loaded_config: Dict) -> Dict:
    """
    Recursively merges two dictionaries. Values from loaded_config override
    values from default_config.
    """
    merged = default_config.copy()
    if loaded_config:
        for key, value in loaded_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

def generate_config_hash(config_dict: Dict, length: int = 7) -> str:
    """Generates a hash from a dictionary, ensuring consistent order."""
    config_string = json.dumps(convert_to_native_types(config_dict), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def convert_to_native_types(data):
    """Converts numpy types within a dict/list to native Python types."""
    if isinstance(data, list): return [convert_to_native_types(item) for item in data]
    if isinstance(data, dict): return {key: convert_to_native_types(value) for key, value in data.items()}
    if isinstance(data, np.integer): return int(data)
    if isinstance(data, np.floating): return float(data)
    if isinstance(data, np.ndarray): return data.tolist() 
    if isinstance(data, (np.bool_, bool)): return bool(data)
    if isinstance(data, pd.Timestamp): return data.isoformat()
    return data

def get_relevant_config_for_hash(effective_config: Dict) -> Dict:
    """
    Extracts relevant configuration parameters for hashing from the effective_config.
    Dynamically includes agent-specific parameters based on 'agent_type'.
    """
    relevant_config_for_hash = {}
    hash_keys_structure = effective_config.get("hash_config_keys", {})

    # Environment keys
    if "environment" in hash_keys_structure and isinstance(hash_keys_structure["environment"], list):
        env_keys_to_hash = hash_keys_structure["environment"]
        relevant_config_for_hash["environment"] = {
            k: effective_config["environment"].get(k) for k in env_keys_to_hash if k in effective_config["environment"]
        }

    # Agent parameters (dynamic based on agent_type)
    agent_type = effective_config.get("agent_type")
    if agent_type and "agent_params" in hash_keys_structure and isinstance(hash_keys_structure["agent_params"], dict):
        if agent_type in hash_keys_structure["agent_params"] and isinstance(hash_keys_structure["agent_params"][agent_type], list):
            agent_keys_to_hash = hash_keys_structure["agent_params"][agent_type]
            algo_params_section = effective_config.get(f"{agent_type.lower()}_params", {}) # e.g., ppo_params
            
            relevant_agent_params = {}
            for k in agent_keys_to_hash:
                if k in algo_params_section:
                    value = algo_params_section.get(k)
                    # Special handling for 'policy_kwargs' string to dict conversion for hashing consistency
                    if k == "policy_kwargs" and isinstance(value, str):
                        try:
                            value = eval(value)
                        except Exception:
                            pass # If eval fails, hash the string as is
                    relevant_agent_params[k] = value
            relevant_config_for_hash["agent_params"] = {agent_type: relevant_agent_params}

    # Binance settings keys
    if "binance_settings" in hash_keys_structure and isinstance(hash_keys_structure["binance_settings"], list):
        bn_keys_to_hash = hash_keys_structure["binance_settings"]
        relevant_config_for_hash["binance_settings"] = {
            k: effective_config["binance_settings"].get(k) for k in bn_keys_to_hash if k in effective_config["binance_settings"]
        }
    
    return {k: v for k, v in relevant_config_for_hash.items() if v}


def resolve_model_path(effective_config: Dict, log_level: str = "normal") -> tuple:
    """
    Resolves the path to the trained model.
    Checks explicit path first, then attempts to reconstruct from training config hash.
    Returns (resolved_model_path, alt_model_path_info).
    """
    run_settings = effective_config.get("run_settings", {})
    
    model_load_path = run_settings.get("model_path")
    alt_model_load_path = run_settings.get("alt_model_path") # This can be used for secondary models if needed

    if model_load_path and os.path.exists(model_load_path) and model_load_path.endswith(".zip"):
        if log_level in ["normal", "detailed"]: print(f"Using explicit model_path: {model_load_path}")
        return model_load_path, alt_model_load_path
    elif model_load_path: 
         if log_level != "none": print(f"Warning: Explicit model_path '{model_load_path}' not found or invalid. Attempting reconstruction.")

    if log_level in ["normal", "detailed"]: print("Attempting to reconstruct model path from training config hash...")
    
    relevant_config_for_hash = get_relevant_config_for_hash(effective_config)

    train_log_dir_base = run_settings.get("log_dir_base")
    train_base_model_name = run_settings.get("model_name")

    if train_log_dir_base and train_base_model_name:
        if not relevant_config_for_hash and log_level != "none": print("Warning: Cannot auto-find model, relevant config for hash is empty.")
        else:
            config_hash = generate_config_hash(relevant_config_for_hash)
            final_model_name_with_hash = f"{config_hash}_{train_base_model_name}"
            # Log directory for models is within logs/training/
            expected_run_dir = os.path.join("logs", "training", final_model_name_with_hash)
            
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