# src/data/utils.py
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
from tqdm import tqdm
import sys # Added for sys.stdout.flush()

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

DATA_CACHE_DIR = "data_cache/"
RANGE_CACHE_SUBDIR = "range_cache"

def _load_single_yaml_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
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
    if default_config_paths is None:
        default_config_paths = []
    merged_config = {}
    for path in default_config_paths:
        default_cfg = _load_single_yaml_config(path)
        merged_config = merge_configs(merged_config, default_cfg)
    main_cfg = _load_single_yaml_config(main_config_path)
    merged_config = merge_configs(merged_config, main_cfg)
    return merged_config

def _calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: list) -> pd.DataFrame:
    df_processed = df.copy()
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Cannot calculate TAs.")
            df_processed[col] = np.nan
            df_processed.fillna(0, inplace=True)
            return df_processed
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.dropna(subset=['High', 'Low', 'Close'], inplace=True)
    if df_processed.empty:
        print("WARNING: DataFrame became empty after dropping NaNs for TA calculation.")
        return df_processed

    if not TALIB_AVAILABLE:
        print("TA-Lib not available, skipping technical indicator calculation.")
        final_df = df_processed[[col for col in required_cols_for_ta if col in df_processed.columns]].copy()
        return final_df.bfill().ffill().fillna(0)

    high_np = df_processed['High'].values.astype(float)
    low_np = df_processed['Low'].values.astype(float)
    close_np = df_processed['Close'].values.astype(float)
    open_np = df_processed['Open'].values.astype(float)
    volume_np = df_processed['Volume'].values.astype(float)


    for feature_name in price_features_to_add:
        if feature_name in required_cols_for_ta:
            continue
        try:
            if feature_name.startswith('SMA_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.SMA(close_np, timeperiod=timeperiod)
            elif feature_name.startswith('EMA_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.EMA(close_np, timeperiod=timeperiod)
            elif feature_name.startswith('RSI_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.RSI(close_np, timeperiod=timeperiod)
            elif feature_name == 'MACD':
                macd, macdsignal, macdhist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
                df_processed['MACD'] = macd
            elif feature_name == 'ADX':
                df_processed['ADX'] = talib.ADX(high_np, low_np, close_np, timeperiod=14)
            elif feature_name == 'STOCH_K':
                stoch_k, stoch_d = talib.STOCH(high_np, low_np, close_np, fastk_period=5, slowk_period=3, slowd_period=3)
                df_processed['STOCH_K'] = stoch_k
            elif feature_name == 'ATR':
                df_processed['ATR'] = talib.ATR(high_np, low_np, close_np, timeperiod=14)
            elif feature_name == 'BBANDS_Upper':
                upper, middle, lower = talib.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                df_processed['BBANDS_Upper'] = upper
            elif feature_name == 'AD':
                df_processed['AD'] = talib.AD(high_np, low_np, close_np, volume_np)
            elif feature_name == 'OBV':
                df_processed['OBV'] = talib.OBV(close_np, volume_np)
            elif feature_name.startswith('CDL'):
                if hasattr(talib, feature_name):
                    pattern_func = getattr(talib, feature_name)
                    if feature_name in ['CDLMORNINGSTAR', 'CDLEVENINGSTAR']:
                         df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np, penetration=0)
                    else:
                         df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np)
                else:
                    print(f"Warning: TA-Lib function '{feature_name}' not found. Assigning NaN.")
                    df_processed[feature_name] = np.nan
            else:
                print(f"Warning: TA '{feature_name}' not defined for calculation. Assigning NaN.")
                df_processed[feature_name] = np.nan
        except Exception as e:
            print(f"Error calculating TA '{feature_name}': {e}. Assigning NaN.")
            df_processed[feature_name] = np.nan

    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True)

    final_columns = [col for col in required_cols_for_ta if col in df_processed.columns]
    for feature in price_features_to_add:
        if feature not in final_columns:
            if feature not in df_processed.columns:
                df_processed[feature] = 0.0
            final_columns.append(feature)

    return df_processed[final_columns]


def fetch_and_cache_kline_data(
    symbol: str, interval: str, start_date_str: str, end_date_str: str,
    cache_dir: str,
    price_features_to_add: list = None,
    api_key: str = None, api_secret: str = None, testnet: bool = False,
    cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2, pbar_instance = None
) -> pd.DataFrame:

    _print_fn = pbar_instance.write if pbar_instance else print

    if not BINANCE_CLIENT_AVAILABLE:
        _print_fn("CRITICAL ERROR in fetch_and_cache_kline_data: python-binance library not found.")
        return pd.DataFrame()

    ta_features_for_filename = sorted([f for f in (price_features_to_add or []) if f not in ['Open', 'High', 'Low', 'Close', 'Volume']])
    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")
    cache_file = get_data_path_for_day(
        date_str=daily_file_date_str, symbol=symbol, data_type="kline",
        interval=interval, price_features_to_add=ta_features_for_filename, cache_dir=cache_dir
    )

    # Ensure base cache directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if log_level == "detailed":
        _print_fn(f"DEBUG_KLINE_DAILY: Checking for daily K-line cache: {cache_file}"); sys.stdout.flush()

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: _print_fn(f"Loading K-line data from daily cache: {cache_file}"); sys.stdout.flush()
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')
            missing_tas = [ta for ta in ta_features_for_filename if ta not in df.columns]
            if missing_tas:
                if log_level != "none": _print_fn(f"Warning: Cached K-line data {cache_file} missing TAs: {missing_tas}. Refetching for this day."); sys.stdout.flush()
                if log_level == "detailed": _print_fn(f"DEBUG_KLINE_DAILY: Removing daily K-line cache due to missing TAs: {cache_file}"); sys.stdout.flush()
                os.remove(cache_file)
            else:
                if log_level == "detailed": _print_fn(f"DEBUG_KLINE_DAILY: Daily K-line cache HIT and valid: {cache_file}, Shape: {df.shape}"); sys.stdout.flush()
                return df
        except Exception as e:
            if log_level != "none": _print_fn(f"Error loading K-line data from daily cache {cache_file}: {e}. Refetching for this day."); sys.stdout.flush()
            if log_level == "detailed": _print_fn(f"DEBUG_KLINE_DAILY: Removing daily K-line cache due to load error: {cache_file}"); sys.stdout.flush()
            if os.path.exists(cache_file): os.remove(cache_file)

    if log_level == "detailed":
        _print_fn(f"DEBUG_KLINE_DAILY: Daily K-line cache MISS or invalid for: {cache_file}. Fetching from API."); sys.stdout.flush()
    if log_level in ["normal", "detailed"]:
        _print_fn(f"Fetching K-line data for {symbol}, Interval: {interval}, Date: {daily_file_date_str} (API range: {start_date_str} to {end_date_str})"); sys.stdout.flush()

    client = Client(api_key or os.environ.get('BINANCE_API_KEY'),
                    api_secret or os.environ.get('BINANCE_API_SECRET'),
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    try:
        klines_raw = client.get_historical_klines(symbol, interval, start_date_str, end_str=end_date_str)
        if not klines_raw:
            if log_level != "none": _print_fn(f"No k-lines returned by API for {symbol} {start_date_str}-{end_date_str} (interval {interval})."); sys.stdout.flush()
            empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'] + ta_features_for_filename)
            empty_df.index = pd.to_datetime([]).tz_localize('UTC')
            if cache_file_type == "parquet": empty_df.to_parquet(cache_file)
            else: empty_df.to_csv(cache_file)
            if log_level == "detailed": _print_fn(f"DEBUG_KLINE_DAILY: Saved empty daily K-line cache: {cache_file}"); sys.stdout.flush()
            return empty_df

        kline_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume',
                        'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore']
        df_fetched = pd.DataFrame(klines_raw, columns=kline_cols)
        df_fetched['OpenTimeDate'] = pd.to_datetime(df_fetched['OpenTime'], unit='ms', utc=True)
        df_fetched.set_index('OpenTimeDate', inplace=True)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df_fetched.columns: df_fetched[col] = pd.to_numeric(df_fetched[col], errors='coerce')
        df_to_process = df_fetched[numeric_cols].copy()
        df_to_process = df_to_process.astype(float)
        df_with_ta = _calculate_technical_indicators(df_to_process, price_features_to_add or [])

        if not df_with_ta.empty:
            try:
                if cache_file_type == "parquet": df_with_ta.to_parquet(cache_file)
                else: df_with_ta.to_csv(cache_file)
                if log_level in ["normal", "detailed"]: _print_fn(f"Daily K-line data with TAs saved to cache: {cache_file}"); sys.stdout.flush()
                if log_level == "detailed": _print_fn(f"DEBUG_KLINE_DAILY: Saved daily K-line data to cache: {cache_file}, Shape: {df_with_ta.shape}"); sys.stdout.flush()
            except Exception as e: _print_fn(f"Error saving daily K-line data to cache {cache_file}: {e}"); sys.stdout.flush()
        return df_with_ta
    except BinanceAPIException as bae:
        _print_fn(f"Binance API Exception during K-line fetch for {daily_file_date_str}: Code={bae.code}, Message='{bae.message}'"); sys.stdout.flush()
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'] + ta_features_for_filename).set_index(pd.to_datetime([]).tz_localize('UTC'))
    except Exception as e:
        _print_fn(f"Unexpected error during K-line fetch/processing for {daily_file_date_str}: {e}"); sys.stdout.flush()
        traceback.print_exc()
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'] + ta_features_for_filename).set_index(pd.to_datetime([]).tz_localize('UTC'))

def fetch_continuous_aggregate_trades(
    symbol: str, start_date_str: str, end_date_str: str,
    cache_dir: str, api_key: str = None, api_secret: str = None,
    testnet: bool = False, cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2, pbar_instance = None
) -> pd.DataFrame:

    _print_fn = pbar_instance.write if pbar_instance else print

    if not BINANCE_CLIENT_AVAILABLE:
        _print_fn("CRITICAL ERROR in fetch_continuous_aggregate_trades: python-binance library not found."); sys.stdout.flush()
        return pd.DataFrame()

    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")
    cache_file = get_data_path_for_day(daily_file_date_str, symbol, data_type="agg_trades", cache_dir=cache_dir)

    # Ensure base cache directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if log_level == "detailed":
        _print_fn(f"DEBUG_AGGTRADES_DAILY: Checking for daily RAW agg_trades cache: {cache_file}"); sys.stdout.flush()

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: _print_fn(f"Loading aggregate trades from daily cache: {cache_file}"); sys.stdout.flush()
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')
            if log_level == "detailed": _print_fn(f"DEBUG_AGGTRADES_DAILY: Daily RAW agg_trades cache HIT and loaded: {cache_file}, Shape: {df.shape}"); sys.stdout.flush()
            return df
        except Exception as e:
            if log_level != "none": _print_fn(f"Error loading aggregate trades from daily cache {cache_file}: {e}. Refetching for this day."); sys.stdout.flush()
            if log_level == "detailed": _print_fn(f"DEBUG_AGGTRADES_DAILY: Removing daily RAW agg_trades cache due to load error: {cache_file}"); sys.stdout.flush()
            if os.path.exists(cache_file): os.remove(cache_file)

    if log_level == "detailed":
        _print_fn(f"DEBUG_AGGTRADES_DAILY: Daily RAW agg_trades cache MISS or invalid for: {cache_file}. Fetching from API."); sys.stdout.flush()
    if log_level in ["normal", "detailed"]: # This condition was missing before for the fetch message itself
        _print_fn(f"Fetching continuous aggregate trades for {symbol} from {start_date_str} to {end_date_str}."); sys.stdout.flush()


    client = Client(api_key or os.environ.get('BINANCE_API_KEY'),
                    api_secret or os.environ.get('BINANCE_API_SECRET'),
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    all_trades = []
    start_dt = pd.to_datetime(start_date_str, utc=True)
    end_dt = pd.to_datetime(end_date_str, utc=True)
    current_start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Use external pbar if provided for its .write method, but manage progress locally for this fetch
    # This local_pbar is for visual feedback of THIS specific fetch operation.
    local_fetch_pbar = tqdm(total=max(1, end_ms - current_start_ms),
                            desc=f"Fetching {symbol} for {daily_file_date_str}",
                            unit="ms", unit_scale=True, leave=False, # leave=False if pbar_instance is also used by outer loop
                            disable=(log_level == "none")) # Disable if no logging


    while current_start_ms < end_ms:
        chunk_end_ms = min(current_start_ms + (60 * 60 * 1000 - 1), end_ms)
        try:
            trades_chunk = client.get_aggregate_trades(
                symbol=symbol, startTime=current_start_ms, endTime=chunk_end_ms, limit=1000
            )
            if trades_chunk:
                all_trades.extend(trades_chunk)
                new_start_ms = trades_chunk[-1]['T'] + 1
            else:
                new_start_ms = chunk_end_ms + 1

            local_fetch_pbar.update(new_start_ms - current_start_ms)
            current_start_ms = new_start_ms
            if current_start_ms >= end_ms: break
            time.sleep(api_request_delay_seconds)
        except BinanceAPIException as bae:
            _print_fn(f"Binance API Exception (agg trades) for {daily_file_date_str}: {bae.code} - {bae.message}. Retrying or stopping..."); sys.stdout.flush()
            time.sleep(max(api_request_delay_seconds * 5, 1))
            if bae.code == -1003:
                _print_fn("Rate limit hit, sleeping for 60s..."); sys.stdout.flush()
                time.sleep(60)
            else: break
        except Exception as e:
            _print_fn(f"Error fetching aggregate trades for {daily_file_date_str}: {e}"); sys.stdout.flush()
            traceback.print_exc()
            break

    if local_fetch_pbar.total > local_fetch_pbar.n : local_fetch_pbar.total = local_fetch_pbar.n
    local_fetch_pbar.close()

    expected_cols_tick = ['Price', 'Quantity', 'IsBuyerMaker']
    if not all_trades:
        if log_level != "none": _print_fn(f"No aggregate trades returned by API for {symbol} {start_date_str}-{end_date_str}."); sys.stdout.flush()
        empty_df = pd.DataFrame(columns=expected_cols_tick)
        empty_df.index = pd.to_datetime([]).tz_localize('UTC')
        if cache_file_type == "parquet": empty_df.to_parquet(cache_file)
        else: empty_df.to_csv(cache_file)
        if log_level == "detailed": _print_fn(f"DEBUG_AGGTRADES_DAILY: Saved empty daily RAW agg_trades cache: {cache_file}"); sys.stdout.flush()
        return empty_df

    df_trades = pd.DataFrame(all_trades)
    df_trades.rename(columns={'T': 'Timestamp', 'p': 'Price', 'q': 'Quantity', 'm': 'IsBuyerMaker'}, inplace=True)
    df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'], unit='ms', utc=True)
    df_trades.set_index('Timestamp', inplace=True)
    df_trades[['Price', 'Quantity']] = df_trades[['Price', 'Quantity']].astype(float)
    df_trades_final = df_trades[expected_cols_tick].copy()

    if not df_trades_final.empty:
        try:
            if cache_file_type == "parquet": df_trades_final.to_parquet(cache_file)
            else: df_trades_final.to_csv(cache_file)
            if log_level in ["normal", "detailed"]: _print_fn(f"Daily aggregate trades saved to cache: {cache_file}"); sys.stdout.flush()
            if log_level == "detailed": _print_fn(f"DEBUG_AGGTRADES_DAILY: Saved daily RAW agg_trades to cache: {cache_file}, Shape: {df_trades_final.shape}"); sys.stdout.flush()
        except Exception as e: _print_fn(f"Error saving daily aggregate trades to cache {cache_file}: {e}"); sys.stdout.flush()
    return df_trades_final

def get_data_path_for_day(date_str: str, symbol: str, data_type: str = "agg_trades",
                          interval: str = None, price_features_to_add: list = None,
                          cache_dir: str = DATA_CACHE_DIR, resample_interval_ms: int = None) -> str:
    # This function now only returns the path string. Directory creation is handled by the saving function.
    symbol_cache_dir = os.path.join(cache_dir, symbol)

    if data_type == "agg_trades":
        filename_prefix = "bn_aggtrades"
        resample_suffix = f"_R{resample_interval_ms}ms" if resample_interval_ms else ""
        safe_filename = f"{filename_prefix}_{symbol}_{date_str}{resample_suffix}.parquet"
    elif data_type == "kline":
        if not interval: raise ValueError("Interval must be provided for kline data type.")
        sorted_features_str = ""
        if price_features_to_add:
            normalized_features_for_filename = sorted([re.sub(r'[^a-zA-Z0-9]', '', f).lower() for f in price_features_to_add])
            sorted_features_str = "_".join(normalized_features_for_filename)
            if sorted_features_str: sorted_features_str = f"_{sorted_features_str}"
        filename_prefix = "bn_klines"
        safe_filename = f"{filename_prefix}_{symbol}_{interval}_{date_str}{sorted_features_str}.parquet"
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Must be 'agg_trades' or 'kline'.")
    return os.path.join(symbol_cache_dir, safe_filename)

def _generate_data_config_hash_key(params: dict, length: int = 10) -> str:
    if 'price_features' in params and isinstance(params['price_features'], list):
        params['price_features'] = sorted(params['price_features'])
    config_string = json.dumps(convert_to_native_types(params), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def _get_range_cache_path(symbol: str, start_date_str: str, end_date_str: str, data_type_suffix: str,
                          cache_config_params: dict, cache_dir: str = DATA_CACHE_DIR) -> str:
    # This function now only returns the path string. Directory creation is handled by the saving function.
    symbol_cache_dir = os.path.join(cache_dir, symbol)
    range_cache_dir = os.path.join(symbol_cache_dir, RANGE_CACHE_SUBDIR)
    config_hash = _generate_data_config_hash_key(cache_config_params)
    safe_start = start_date_str.replace(" ", "_").replace(":", "")
    safe_end = end_date_str.replace(" ", "_").replace(":", "")
    filename = f"{config_hash}_{symbol}_{data_type_suffix}_{safe_start}_to_{safe_end}.parquet"
    return os.path.join(range_cache_dir, filename)

def fetch_and_cache_tick_data(*args, **kwargs):
    return fetch_continuous_aggregate_trades(*args, **kwargs)


def load_tick_data_for_range(symbol: str, start_date_str: str, end_date_str: str, cache_dir: str = DATA_CACHE_DIR,
                             binance_settings: Dict = None, tick_resample_interval_ms: int = None,
                             log_level: str = "normal") -> pd.DataFrame:

    if log_level != "none": print(f"[[load_tick_data_for_range ENTRY]] Log level received: {log_level}"); sys.stdout.flush()

    if binance_settings is None: binance_settings = {}

    range_cache_config = {
        "symbol": symbol, "start_date": start_date_str, "end_date": end_date_str,
        "type": "ticks", "resample_ms": tick_resample_interval_ms
    }
    range_cache_file_path = _get_range_cache_path(symbol, start_date_str, end_date_str, "ticks",
                                                  range_cache_config, cache_dir)

    if log_level == "detailed":
        print(f"DEBUG_LOAD_TICK (Range): Attempting to load from full range cache: {range_cache_file_path}"); sys.stdout.flush()

    if os.path.exists(range_cache_file_path):
        try:
            if log_level != "none": print(f"Loading FULL RANGE tick data from cache: {range_cache_file_path}"); sys.stdout.flush()
            df_combined = pd.read_parquet(range_cache_file_path)
            if not df_combined.empty and isinstance(df_combined.index, pd.DatetimeIndex):
                 if df_combined.index.tz is None: df_combined.index = df_combined.index.tz_localize('UTC')
                 if log_level == "detailed": print(f"DEBUG_LOAD_TICK (Range): Full range cache HIT. Shape: {df_combined.shape}"); sys.stdout.flush()
                 return df_combined
            else:
                 if log_level != "none": print(f"Warning: Range cache file {range_cache_file_path} is empty or invalid. Re-processing."); sys.stdout.flush()
                 if log_level == "detailed": print(f"DEBUG_LOAD_TICK (Range): Removing invalid range cache: {range_cache_file_path}"); sys.stdout.flush()
                 if os.path.exists(range_cache_file_path): os.remove(range_cache_file_path)
        except Exception as e:
            if log_level != "none": print(f"Error loading from range cache {range_cache_file_path}: {e}. Re-processing."); sys.stdout.flush()
            if log_level == "detailed": print(f"DEBUG_LOAD_TICK (Range): Removing range cache due to load error: {range_cache_file_path}"); sys.stdout.flush()
            if os.path.exists(range_cache_file_path): os.remove(range_cache_file_path)
    elif log_level == "detailed":
        print(f"DEBUG_LOAD_TICK (Range): Full range cache MISS: {range_cache_file_path}"); sys.stdout.flush()

    start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
    all_data_frames = []
    current_date_obj = start_date_obj

    pbar_days = tqdm(total=(end_date_obj - start_date_obj).days + 1, desc=f"Processing Ticks {symbol}", leave=True)

    while current_date_obj <= end_date_obj:
        date_str_for_day = current_date_obj.strftime('%Y-%m-%d')
        _print_fn = pbar_days.write

        if log_level != "none": _print_fn(f"[[load_tick_data_for_range DAILY_LOOP]] Day: {date_str_for_day}, Current log_level: {log_level}"); sys.stdout.flush()

        if log_level == "detailed": _print_fn(f"\nDEBUG_LOAD_TICK (Daily): --- Processing day: {date_str_for_day} ---"); sys.stdout.flush()

        daily_file_path = get_data_path_for_day(date_str_for_day, symbol, data_type="agg_trades",
                                                cache_dir=cache_dir, resample_interval_ms=tick_resample_interval_ms)
        df_daily = pd.DataFrame()

        if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Checking for daily RESAMPLED file: {daily_file_path}"); sys.stdout.flush()
        if os.path.exists(daily_file_path):
            if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Daily RESAMPLED file EXISTS. Attempting to load."); sys.stdout.flush()
            try:
                df_daily = pd.read_parquet(daily_file_path)
                if df_daily.index.tz is None and not df_daily.empty: df_daily.index = df_daily.index.tz_localize('UTC')
                if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Loaded daily RESAMPLED file {daily_file_path}, shape: {df_daily.shape if not df_daily.empty else 'EMPTY'}"); sys.stdout.flush()
            except Exception as e:
                if log_level != "none": _print_fn(f"Error loading daily RESAMPLED file {daily_file_path}: {e}. Will try to generate from raw."); sys.stdout.flush()
                if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Removing corrupted daily RESAMPLED file: {daily_file_path}"); sys.stdout.flush()
                if os.path.exists(daily_file_path): os.remove(daily_file_path)
                df_daily = pd.DataFrame()
        elif log_level == "detailed":
            _print_fn(f"DEBUG_LOAD_TICK (Daily): Daily RESAMPLED file NOT found: {daily_file_path}"); sys.stdout.flush()

        if df_daily.empty:
            if log_level != "none": _print_fn(f"[[load_tick_data_for_range RAW_CHECK_TRIGGERED]] Resampled empty for {date_str_for_day}. Checking RAW."); sys.stdout.flush()
            if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Daily RESAMPLED data for {date_str_for_day} is empty or was not loaded. Attempting to use/fetch RAW."); sys.stdout.flush()

            raw_daily_file_path = get_data_path_for_day(date_str_for_day, symbol, data_type="agg_trades", cache_dir=cache_dir)
            df_raw_daily = pd.DataFrame()

            if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Checking for RAW daily file: {raw_daily_file_path}"); sys.stdout.flush()
            if os.path.exists(raw_daily_file_path):
                if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): RAW daily file EXISTS. Attempting to load."); sys.stdout.flush()
                try:
                    df_raw_daily = pd.read_parquet(raw_daily_file_path)
                    if df_raw_daily.index.tz is None and not df_raw_daily.empty: df_raw_daily.index = df_raw_daily.index.tz_localize('UTC')
                    if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Loaded RAW daily file {raw_daily_file_path}, shape: {df_raw_daily.shape if not df_raw_daily.empty else 'EMPTY'}"); sys.stdout.flush()
                except Exception as e_load_raw:
                    if log_level != "none": _print_fn(f"  WARNING: Could not read existing RAW daily file {raw_daily_file_path} (Error: {e_load_raw}). File will be treated as missing, potentially leading to re-download."); sys.stdout.flush()
                    if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): RAW daily file load FAILED from {raw_daily_file_path}. Error: {e_load_raw}"); sys.stdout.flush()
            elif log_level == "detailed":
                 _print_fn(f"DEBUG_LOAD_TICK (Daily): RAW daily file NOT found: {raw_daily_file_path}"); sys.stdout.flush()

            if df_raw_daily.empty:
                if log_level != "none": _print_fn(f"Missing raw daily data for {symbol} on {date_str_for_day}. Fetching."); sys.stdout.flush()

                day_start_dt_utc = datetime.combine(current_date_obj, datetime.min.time(), tzinfo=timezone.utc)
                day_end_dt_utc = datetime.combine(current_date_obj, datetime.max.time(), tzinfo=timezone.utc)

                df_raw_daily = fetch_continuous_aggregate_trades(
                    symbol=symbol, start_date_str=day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date_str=day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    cache_dir=cache_dir, api_key=binance_settings.get("api_key"),
                    api_secret=binance_settings.get("api_secret"), testnet=binance_settings.get("testnet", False),
                    log_level="detailed" if log_level=="detailed" else "none",
                    api_request_delay_seconds=binance_settings.get("api_request_delay_seconds", 0.2),
                    pbar_instance=pbar_days
                )
                if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Fetched raw daily data for {date_str_for_day}, shape: {df_raw_daily.shape if not df_raw_daily.empty else 'EMPTY'}"); sys.stdout.flush()
            elif log_level == "detailed":
                _print_fn(f"DEBUG_LOAD_TICK (Daily): Using RAW daily data loaded from CACHE for {date_str_for_day}, shape {df_raw_daily.shape if not df_raw_daily.empty else 'EMPTY'}"); sys.stdout.flush()

            if df_raw_daily.empty:
                if log_level != "none": _print_fn(f"No raw tick data available (from cache or fetch) for {symbol} on {date_str_for_day}. Skipping."); sys.stdout.flush()
                current_date_obj += timedelta(days=1)
                pbar_days.update(1)
                continue

            if tick_resample_interval_ms:
                if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Resampling raw data for {date_str_for_day} to {tick_resample_interval_ms}ms. Original shape: {df_raw_daily.shape if not df_raw_daily.empty else 'EMPTY RAW DF'}"); sys.stdout.flush()
                if not isinstance(df_raw_daily.index, pd.DatetimeIndex) or df_raw_daily.index.tz is None:
                     df_raw_daily.index = pd.to_datetime(df_raw_daily.index, utc=True)
                try:
                    freq_str = f"{tick_resample_interval_ms}ms"
                    agg_rules = {'Price': 'last', 'Quantity': 'sum', 'IsBuyerMaker': 'last'}
                    valid_agg_rules = {col: rule for col, rule in agg_rules.items() if col in df_raw_daily.columns}

                    df_daily = df_raw_daily.resample(freq_str).agg(valid_agg_rules)
                    df_daily.ffill(inplace=True)
                    df_daily.bfill(inplace=True)
                    for col, default_val in {'Price': 0, 'Quantity': 0, 'IsBuyerMaker': False}.items():
                        if col not in df_daily.columns: df_daily[col] = default_val
                    df_daily.fillna({'Price': 0, 'Quantity': 0, 'IsBuyerMaker': False}, inplace=True)


                    if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Resampled shape: {df_daily.shape}. Saving to: {daily_file_path}"); sys.stdout.flush()
                    
                    # Ensure directory exists before saving
                    os.makedirs(os.path.dirname(daily_file_path), exist_ok=True)
                    df_daily.to_parquet(daily_file_path)

                    if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Resampled daily data for {date_str_for_day} saved to {daily_file_path}"); sys.stdout.flush()
                except Exception as e_resample:
                    if log_level != "none": _print_fn(f"Error resampling daily tick data for {date_str_for_day}: {e_resample}. Using raw."); sys.stdout.flush()
                    if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Resampling failed. Using raw data for {date_str_for_day}. Error: {e_resample}"); sys.stdout.flush()
                    df_daily = df_raw_daily.copy()
            else:
                if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): No resampling needed. Using raw data for {date_str_for_day}."); sys.stdout.flush()
                df_daily = df_raw_daily.copy()

        if not df_daily.empty:
            if log_level == "detailed": _print_fn(f"DEBUG_LOAD_TICK (Daily): Appending data for day {date_str_for_day}. Shape: {df_daily.shape}"); sys.stdout.flush()
            all_data_frames.append(df_daily)
        elif log_level == "detailed":
            _print_fn(f"DEBUG_LOAD_TICK (Daily): df_daily is empty for {date_str_for_day} before appending. Not appending."); sys.stdout.flush()

        current_date_obj += timedelta(days=1)
        pbar_days.update(1)
    pbar_days.close()

    _print_fn_after_loop = print

    if not all_data_frames:
        if log_level != "none": _print_fn_after_loop(f"No tick data found or loaded for {symbol} in range {start_date_str} to {end_date_str}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames)
    if not combined_df.empty:
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # --- DATETIME FILTERING LOGIC ---
        try:
            start_datetime_utc = pd.to_datetime(start_date_str, utc=True)
            end_datetime_utc = pd.to_datetime(end_date_str, utc=True)
            original_count = len(combined_df)
            combined_df = combined_df.loc[start_datetime_utc:end_datetime_utc]
            if log_level in ["normal", "detailed"] and original_count > 0:
                print(f"Applied precise datetime filter: {original_count} -> {len(combined_df)} rows from {start_datetime_utc} to {end_datetime_utc}")
        except Exception as e_filter:
            if log_level != "none":
                print(f"WARNING: Could not apply precise datetime filter to combined tick data: {e_filter}. Returning daily-aligned data.")
        
        if log_level == "detailed":
            _print_fn_after_loop(f"DEBUG_LOAD_TICK (Range): Combined all daily tick data. Shape: {combined_df.shape}")
        
        try:
            # Ensure directory for range cache exists before saving
            os.makedirs(os.path.dirname(range_cache_file_path), exist_ok=True)
            if log_level != "none": _print_fn_after_loop(f"Saving combined tick data to range cache: {range_cache_file_path}")
            combined_df.to_parquet(range_cache_file_path)
            if log_level == "detailed": _print_fn_after_loop(f"DEBUG_LOAD_TICK (Range): Successfully saved to range cache: {range_cache_file_path}")
        except Exception as e:
            if log_level != "none": _print_fn_after_loop(f"Error saving combined tick data to range cache {range_cache_file_path}: {e}")
    elif log_level == "detailed":
        _print_fn_after_loop(f"DEBUG_LOAD_TICK (Range): No data frames to combine for range cache.")

    return combined_df


def load_kline_data_for_range(symbol: str, start_date_str: str, end_date_str: str, interval: str,
                              price_features: list, cache_dir: str = DATA_CACHE_DIR,
                              binance_settings: Dict = None, log_level: str = "normal") -> pd.DataFrame:
    if log_level != "none": print(f"[[load_kline_data_for_range ENTRY]] Log level received: {log_level}"); sys.stdout.flush()
    if binance_settings is None: binance_settings = {}

    sorted_price_features_for_hash = sorted(price_features or [])
    range_cache_config = {
        "symbol": symbol, "start_date": start_date_str, "end_date": end_date_str,
        "type": "klines", "interval": interval, "features": sorted_price_features_for_hash
    }
    range_cache_file_path = _get_range_cache_path(symbol, start_date_str, end_date_str, f"klines_{interval}",
                                                  range_cache_config, cache_dir)

    _print_fn_range = print

    if log_level == "detailed":
        _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Attempting to load from full range cache: {range_cache_file_path}"); sys.stdout.flush()

    if os.path.exists(range_cache_file_path):
        try:
            if log_level != "none": _print_fn_range(f"Loading FULL RANGE kline data from cache: {range_cache_file_path}"); sys.stdout.flush()
            df_combined = pd.read_parquet(range_cache_file_path)
            if not df_combined.empty and isinstance(df_combined.index, pd.DatetimeIndex):
                if df_combined.index.tz is None: df_combined.index = df_combined.index.tz_localize('UTC')
                if all(feat in df_combined.columns for feat in price_features):
                    if log_level == "detailed": _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Full range cache HIT. Shape: {df_combined.shape}"); sys.stdout.flush()
                    return df_combined
                else:
                    if log_level != "none": _print_fn_range(f"Warning: Range cache file {range_cache_file_path} missing requested features. Re-processing."); sys.stdout.flush()
                    if log_level == "detailed": _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Removing invalid range cache (missing features): {range_cache_file_path}"); sys.stdout.flush()
                    os.remove(range_cache_file_path)
            else:
                 if log_level != "none": _print_fn_range(f"Warning: Range cache file {range_cache_file_path} is empty or invalid. Re-processing."); sys.stdout.flush()
                 if log_level == "detailed": _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Removing invalid range cache (empty/invalid): {range_cache_file_path}"); sys.stdout.flush()
                 os.remove(range_cache_file_path)
        except Exception as e:
            if log_level != "none": _print_fn_range(f"Error loading from range cache {range_cache_file_path}: {e}. Re-processing."); sys.stdout.flush()
            if log_level == "detailed": _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Removing range cache due to load error: {range_cache_file_path}"); sys.stdout.flush()
            if os.path.exists(range_cache_file_path): os.remove(range_cache_file_path)
    elif log_level == "detailed":
        _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Full range cache MISS: {range_cache_file_path}"); sys.stdout.flush()


    start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
    all_data_frames = []
    current_date_obj = start_date_obj

    pbar_days_kline = tqdm(total=(end_date_obj - start_date_obj).days + 1, desc=f"Processing Klines {symbol} {interval}", leave=True)

    while current_date_obj <= end_date_obj:
        date_str_for_day = current_date_obj.strftime('%Y-%m-%d')
        _print_fn_daily = pbar_days_kline.write

        if log_level != "none": _print_fn_daily(f"[[load_kline_data_for_range DAILY_LOOP]] Day: {date_str_for_day}, Current log_level: {log_level}"); sys.stdout.flush()


        if log_level == "detailed": _print_fn_daily(f"\nDEBUG_LOAD_KLINE (Daily): --- Processing day: {date_str_for_day} ---"); sys.stdout.flush()

        day_api_start_str = datetime.combine(current_date_obj, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
        day_api_end_str = datetime.combine(current_date_obj, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")

        df_daily = fetch_and_cache_kline_data(
            symbol=symbol, interval=interval,
            start_date_str=day_api_start_str,
            end_date_str=day_api_end_str,
            cache_dir=cache_dir,
            price_features_to_add=price_features,
            api_key=binance_settings.get("api_key"), api_secret=binance_settings.get("api_secret"),
            testnet=binance_settings.get("testnet", False),
            log_level="detailed" if log_level=="detailed" else "none",
            api_request_delay_seconds=binance_settings.get("api_request_delay_seconds", 0.2),
            pbar_instance=pbar_days_kline
        )

        if df_daily is not None and not df_daily.empty:
            if log_level == "detailed": _print_fn_daily(f"DEBUG_LOAD_KLINE (Daily): Appending K-line data for day {date_str_for_day}. Shape: {df_daily.shape}"); sys.stdout.flush()
            all_data_frames.append(df_daily)
        elif log_level == "detailed":
            _print_fn_daily(f"DEBUG_LOAD_KLINE (Daily): No K-line data for {symbol} on {date_str_for_day} for interval {interval}. Skipping append."); sys.stdout.flush()

        current_date_obj += timedelta(days=1)
        pbar_days_kline.update(1)
    pbar_days_kline.close()

    if not all_data_frames:
        if log_level != "none": _print_fn_range(f"No K-line data found or loaded for {symbol} in range {start_date_str} to {end_date_str} for interval {interval}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames)
    if not combined_df.empty:
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # --- DATETIME FILTERING LOGIC ---
        try:
            start_datetime_utc = pd.to_datetime(start_date_str, utc=True)
            end_datetime_utc = pd.to_datetime(end_date_str, utc=True)
            original_count = len(combined_df)
            combined_df = combined_df.loc[start_datetime_utc:end_datetime_utc]
            if log_level in ["normal", "detailed"] and original_count > 0:
                print(f"Applied precise datetime filter: {original_count} -> {len(combined_df)} rows from {start_datetime_utc} to {end_datetime_utc}")
        except Exception as e_filter:
            if log_level != "none":
                print(f"WARNING: Could not apply precise datetime filter to combined k-line data: {e_filter}. Returning daily-aligned data.")

        for feat in price_features:
            if feat not in combined_df.columns:
                if log_level != "none": _print_fn_range(f"Warning: Feature '{feat}' missing in combined kline data. Filling with 0.")
                combined_df[feat] = 0.0
        try:
            combined_df = combined_df[price_features]
        except KeyError as e:
            if log_level != "none": _print_fn_range(f"Error selecting final columns for K-line data: {e}. Available: {combined_df.columns.tolist()}")
            combined_df = combined_df[[col for col in price_features if col in combined_df.columns]]


        if log_level == "detailed": _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Combined all daily K-line data. Shape: {combined_df.shape}")

        try:
            # Ensure directory for range cache exists before saving
            os.makedirs(os.path.dirname(range_cache_file_path), exist_ok=True)
            if log_level != "none": _print_fn_range(f"Saving combined kline data to range cache: {range_cache_file_path}")
            combined_df.to_parquet(range_cache_file_path)
            if log_level == "detailed": _print_fn_range(f"DEBUG_LOAD_KLINE (Range): Successfully saved to range cache: {range_cache_file_path}")
        except Exception as e:
            if log_level != "none": _print_fn_range(f"Error saving combined kline data to range cache {range_cache_file_path}: {e}")
    elif log_level == "detailed":
        _print_fn_range(f"DEBUG_LOAD_KLINE (Range): No K-line data frames to combine for range cache.")

    return combined_df


def merge_configs(default_config: Dict, loaded_config: Dict) -> Dict:
    merged = default_config.copy()
    if loaded_config:
        for key, value in loaded_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

def generate_config_hash(config_dict: Dict, length: int = 7) -> str:
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

def get_relevant_config_for_hash(effective_config: Dict) -> Dict:
    relevant_config_for_hash = {}
    hash_keys_structure = effective_config.get("hash_config_keys", {})

    # Process run_settings if defined in hash_keys
    if "run_settings" in hash_keys_structure and isinstance(hash_keys_structure["run_settings"], list):
        relevant_config_for_hash["run_settings"] = {
            k: effective_config["run_settings"].get(k) for k in hash_keys_structure["run_settings"] if k in effective_config.get("run_settings",{})
        }

    if "environment" in hash_keys_structure and isinstance(hash_keys_structure["environment"], list):
        relevant_config_for_hash["environment"] = {
            k: effective_config["environment"].get(k) for k in hash_keys_structure["environment"] if k in effective_config.get("environment",{})
        }
    agent_type = effective_config.get("agent_type")
    if agent_type and "agent_params" in hash_keys_structure and isinstance(hash_keys_structure["agent_params"], dict):
        if agent_type in hash_keys_structure["agent_params"] and isinstance(hash_keys_structure["agent_params"][agent_type], list):
            agent_keys_to_hash = hash_keys_structure["agent_params"][agent_type]
            algo_params_section_name = f"{agent_type.lower()}_params"
            algo_params_section = effective_config.get(algo_params_section_name, {})
            relevant_agent_params = {}
            for k in agent_keys_to_hash:
                if k in algo_params_section:
                    value = algo_params_section.get(k)
                    if k == "policy_kwargs" and isinstance(value, str):
                        try: value = eval(value)
                        except: pass
                    relevant_agent_params[k] = value
            if relevant_agent_params:
                 relevant_config_for_hash[algo_params_section_name] = relevant_agent_params

    if "binance_settings" in hash_keys_structure and isinstance(hash_keys_structure["binance_settings"], list):
        relevant_config_for_hash["binance_settings"] = {
            k: effective_config["binance_settings"].get(k) for k in hash_keys_structure["binance_settings"] if k in effective_config.get("binance_settings",{})
        }
    return {k: v for k, v in relevant_config_for_hash.items() if v}


def resolve_model_path(effective_config: Dict, log_level: str = "normal") -> tuple:
    run_settings = effective_config.get("run_settings", {})
    model_load_path = run_settings.get("model_path")
    alt_model_load_path = run_settings.get("alt_model_path")

    if model_load_path and os.path.exists(model_load_path) and model_load_path.endswith(".zip"):
        if log_level in ["normal", "detailed"]: print(f"Using explicit model_path: {model_load_path}")
        return model_load_path, alt_model_load_path
    elif model_load_path and log_level != "none":
         print(f"Warning: Explicit model_path '{model_load_path}' not found or invalid. Attempting reconstruction.")

    if log_level in ["normal", "detailed"]: print("Attempting to reconstruct model path from training config hash...")
    relevant_config_for_hash = get_relevant_config_for_hash(effective_config)
    train_log_dir_base = run_settings.get("log_dir_base", "logs/")


    train_base_model_name = run_settings.get("model_name")
    if train_log_dir_base and train_base_model_name:
        if not relevant_config_for_hash and log_level != "none":
            print("Warning: Cannot auto-find model, relevant config for hash is empty.")
        else:
            config_hash = generate_config_hash(relevant_config_for_hash)
            final_model_name_with_hash = f"{config_hash}_{train_base_model_name}"

            if "training" not in train_log_dir_base.lower().replace("\\", "/").split("/"):
                 expected_run_dir_base_for_training = os.path.join(train_log_dir_base, "training")
            else:
                 expected_run_dir_base_for_training = train_log_dir_base

            expected_run_dir = os.path.join(expected_run_dir_base_for_training, final_model_name_with_hash)

            if log_level == "detailed":
                print(f"Expected run directory for model (based on current config for hash): {expected_run_dir}")
                print(f"  Relevant parts for hash were: {json.dumps(convert_to_native_types(relevant_config_for_hash), indent=2, sort_keys=True)}")

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