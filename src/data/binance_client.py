# src/data/binance_client.py
import os
import pandas as pd
import time
import sys
import traceback
from tqdm import tqdm
# Import the feature engineer function
from src.data.feature_engineer import calculate_technical_indicators
# Import the new, generic hash generation function and the path generator
from src.data.path_manager import get_data_path_for_day, generate_config_hash # Changed: Using generate_config_hash

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    print("WARNING: python-binance library not found. Binance API functions will be unavailable.")

def fetch_and_cache_kline_data(
    symbol: str,
    interval: str,
    start_date_str: str,
    end_date_str: str,
    cache_dir: str,
    # New parameter: dictionary of technical indicator configurations
    technical_indicators_config: dict,
    api_key: str = None,
    api_secret: str = None,
    testnet: bool = False,
    cache_file_type: str = "parquet",
    log_level: str = "normal",
    api_request_delay_seconds: float = 0.2,
    pbar_instance = None # tqdm progress bar instance for external control
) -> pd.DataFrame:
    """
    Fetches K-line (candlestick) data for a given symbol and interval from Binance,
    calculates specified technical indicators, and caches the result.
    If a cached file with the exact configuration (identified by kline_config_hash)
    exists, it loads the data from cache.

    Args:
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        interval (str): K-line interval (e.g., "1m", "1h", "1d").
        start_date_str (str): The start date for fetching data (e.g., "2023-01-01").
        end_date_str (str): The end date for fetching data (e.g., "2023-01-02").
        cache_dir (str): Base directory for data caching.
        technical_indicators_config (dict): A dictionary defining the technical
                                            indicators to calculate as features.
                                            This is crucial for generating a unique cache hash.
        api_key (str, optional): Binance API key. Defaults to environment variable.
        api_secret (str, optional): Binance API secret. Defaults to environment variable.
        testnet (bool, optional): Whether to use Binance testnet. Defaults to False.
        cache_file_type (str, optional): Type of file to use for caching ("parquet" or "csv").
                                        Defaults to "parquet".
        log_level (str, optional): Logging verbosity ("normal", "detailed", "none").
                                   Defaults to "normal".
        api_request_delay_seconds (float, optional): Delay between API requests to
                                                     respect rate limits. Defaults to 0.2.
        pbar_instance (tqdm.tqdm, optional): An external tqdm progress bar instance.

    Returns:
        pd.DataFrame: A DataFrame containing K-line data with calculated technical indicators.
                      Returns an empty DataFrame on error or no data.
    """
    _print_fn = print # Always use print for logging

    if not BINANCE_CLIENT_AVAILABLE:
        _print_fn("CRITICAL ERROR: python-binance library not found. Cannot fetch K-line data.")
        return pd.DataFrame()

    # Calculate a hash for the technical indicators configuration.
    # This hash will be part of the filename to ensure unique caches for different TA sets.
    kline_config_hash = generate_config_hash(technical_indicators_config)

    # Determine the daily file date string, typically based on the start_date_str for the daily cache granularity
    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")

    # Get the full path for the potential cached file
    cache_file = get_data_path_for_day(
        date_str=daily_file_date_str,
        symbol=symbol,
        data_type="kline",
        interval=interval,
        cache_dir=cache_dir,
        kline_config_hash=kline_config_hash # Pass the generated hash to path_manager
    )

    # Ensure the directory for the cache file exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Prepare a list of all expected columns (OHLCV + TA features) for empty dataframes
    base_kline_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Get feature names from the keys of the technical_indicators_config dictionary
    ta_feature_names = list(technical_indicators_config.keys())
    all_expected_cols = base_kline_cols + ta_feature_names

    if log_level == "detailed":
        _print_fn(f"DEBUG_KLINE_DAILY: Checking for daily K-line cache: {cache_file}"); sys.stdout.flush()

    # Check if the cached file exists and try to load it
    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]:
            _print_fn(f"Loading K-line data from daily cache: {cache_file}"); sys.stdout.flush()
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure the index is timezone-aware if not already
            if df.index.tz is None and not df.empty:
                df.index = df.index.tz_localize('UTC')

            # Basic validation: check if expected columns are present.
            # No need to check for missing TAs specifically if the hash already implies their presence.
            # If the file exists and loads, and the hash matches, it's considered valid.
            # Any missing columns would indicate a corrupted cache or a change in the expected features
            # not reflected in the hash, in which case we refetch.
            if not all(col in df.columns for col in all_expected_cols):
                 if log_level != "none":
                     _print_fn(f"Warning: Cached K-line data {cache_file} missing expected columns. Refetching for this day."); sys.stdout.flush()
                 if log_level == "detailed":
                     _print_fn(f"DEBUG_KLINE_DAILY: Removing daily K-line cache due to missing columns: {cache_file}"); sys.stdout.flush()
                 os.remove(cache_file) # Remove invalid cache to force refetch
            else:
                if log_level == "detailed":
                    _print_fn(f"DEBUG_KLINE_DAILY: Daily K-line cache HIT and valid: {cache_file}, Shape: {df.shape}"); sys.stdout.flush()
                return df
        except Exception as e:
            # If there's any error loading the cache, log it and remove the corrupted file
            if log_level != "none":
                _print_fn(f"Error loading K-line data from daily cache {cache_file}: {e}. Refetching for this day."); sys.stdout.flush()
            if log_level == "detailed":
                _print_fn(f"DEBUG_KLINE_DAILY: Removing daily K-line cache due to load error: {cache_file}"); sys.stdout.flush()
            if os.path.exists(cache_file):
                os.remove(cache_file)

    # If cache not found or invalid, fetch from API
    if log_level == "detailed":
        _print_fn(f"DEBUG_KLINE_DAILY: Daily K-line cache MISS or invalid for: {cache_file}. Fetching from API."); sys.stdout.flush()
    if log_level in ["normal", "detailed"]:
        _print_fn(f"Fetching K-line data for {symbol}, Interval: {interval}, Date: {daily_file_date_str} (API range: {start_date_str} to {end_date_str})"); sys.stdout.flush()

    # Initialize Binance client
    client = Client(api_key or os.environ.get('BINANCE_API_KEY'),
                    api_secret or os.environ.get('BINANCE_API_SECRET'),
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    try:
        # Fetch raw historical klines
        klines_raw = client.get_historical_klines(symbol, interval, start_date_str, end_str=end_date_str)
        if not klines_raw:
            if log_level != "none":
                _print_fn(f"No k-lines returned by API for {symbol} {start_date_str}-{end_date_str} (interval {interval})."); sys.stdout.flush()
            # Return an empty DataFrame with all expected columns if no data is fetched
            empty_df = pd.DataFrame(columns=all_expected_cols)
            empty_df.index = pd.to_datetime([]).tz_localize('UTC')
            # Cache the empty DataFrame to avoid repeated API calls for same missing data
            if cache_file_type == "parquet": empty_df.to_parquet(cache_file)
            else: empty_df.to_csv(cache_file)
            if log_level == "detailed":
                _print_fn(f"DEBUG_KLINE_DAILY: Saved empty daily K-line cache: {cache_file}"); sys.stdout.flush()
            return empty_df

        # Process raw klines into a DataFrame
        kline_raw_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume',
                        'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore']
        df_fetched = pd.DataFrame(klines_raw, columns=kline_raw_cols)
        df_fetched['OpenTimeDate'] = pd.to_datetime(df_fetched['OpenTime'], unit='ms', utc=True)
        df_fetched.set_index('OpenTimeDate', inplace=True)
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df_fetched.columns:
                df_fetched[col] = pd.to_numeric(df_fetched[col], errors='coerce')
        
        # Create a copy with only numeric OHLCV for feature engineering
        df_to_process = df_fetched[numeric_cols].copy()
        df_to_process = df_to_process.astype(float) # Ensure float type for TA-Lib

        # Calculate technical indicators using the new config
        df_with_ta = calculate_technical_indicators(df_to_process, technical_indicators_config)

        if not df_with_ta.empty:
            # Ensure all expected columns are present, fill missing with 0 if TA-Lib failed for some
            for col in all_expected_cols:
                if col not in df_with_ta.columns:
                    df_with_ta[col] = 0.0

            try:
                if cache_file_type == "parquet": df_with_ta.to_parquet(cache_file)
                else: df_with_ta.to_csv(cache_file)
                if log_level in ["normal", "detailed"]:
                    _print_fn(f"Daily K-line data with TAs saved to cache: {cache_file}"); sys.stdout.flush()
                if log_level == "detailed":
                    _print_fn(f"DEBUG_KLINE_DAILY: Saved daily K-line data to cache: {cache_file}, Shape: {df_with_ta.shape}"); sys.stdout.flush()
            except Exception as e:
                _print_fn(f"Error saving daily K-line data to cache {cache_file}: {e}"); sys.stdout.flush()
        else:
            # If df_with_ta becomes empty after processing, save an empty DataFrame
            if log_level != "none":
                _print_fn(f"WARNING: df_with_ta became empty after processing for {daily_file_date_str}. Saving empty DataFrame to cache."); sys.stdout.flush()
            empty_df = pd.DataFrame(columns=all_expected_cols)
            empty_df.index = pd.to_datetime([]).tz_localize('UTC')
            if cache_file_type == "parquet": empty_df.to_parquet(cache_file)
            else: empty_df.to_csv(cache_file)


        return df_with_ta
    except BinanceAPIException as bae:
        _print_fn(f"Binance API Exception during K-line fetch for {daily_file_date_str}: Code={bae.code}, Message='{bae.message}'"); sys.stdout.flush()
        # Return empty DataFrame on API error, with expected columns
        empty_df_on_error = pd.DataFrame(columns=all_expected_cols).set_index(pd.to_datetime([]).tz_localize('UTC'))
        # Optionally, save this empty DataFrame to cache to prevent repeated API calls for persistent errors
        # if cache_file_type == "parquet": empty_df_on_error.to_parquet(cache_file)
        # else: empty_df_on_error.to_csv(cache_file)
        return empty_df_on_error
    except Exception as e:
        _print_fn(f"Unexpected error during K-line fetch/processing for {daily_file_date_str}: {e}"); sys.stdout.flush()
        traceback.print_exc()
        # Return empty DataFrame on unexpected error, with expected columns
        empty_df_on_error = pd.DataFrame(columns=all_expected_cols).set_index(pd.to_datetime([]).tz_localize('UTC'))
        # Optionally, save this empty DataFrame to cache
        # if cache_file_type == "parquet": empty_df_on_error.to_parquet(cache_file)
        # else: empty_df_on_error.to_csv(cache_file)
        return empty_df_on_error

def fetch_continuous_aggregate_trades(
    symbol: str, start_date_str: str, end_date_str: str,
    cache_dir: str, api_key: str = None, api_secret: str = None,
    testnet: bool = False, cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2, pbar_instance = None
) -> pd.DataFrame:
    """
    Fetches raw aggregate trade data from Binance and caches it daily.
    This function handles the fetching and initial caching of raw trade data,
    which can then be resampled into K-lines or other formats.

    Args:
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        start_date_str (str): The start date for fetching data (e.g., "2023-01-01").
        end_date_str (str): The end date for fetching data (e.g., "2023-01-02").
        cache_dir (str): Base directory for data caching.
        api_key (str, optional): Binance API key. Defaults to environment variable.
        api_secret (str, optional): Binance API secret. Defaults to environment variable.
        testnet (bool, optional): Whether to use Binance testnet. Defaults to False.
        cache_file_type (str, optional): Type of file to use for caching ("parquet" or "csv").
                                        Defaults to "parquet".
        log_level (str, optional): Logging verbosity ("normal", "detailed", "none").
                                   Defaults to "normal".
        api_request_delay_seconds (float, optional): Delay between API requests to
                                                     respect rate limits. Defaults to 0.2.
        pbar_instance (tqdm.tqdm, optional): An external tqdm progress bar instance.

    Returns:
        pd.DataFrame: A DataFrame containing raw aggregate trades.
                      Returns an empty DataFrame on error or no data.
    """

    _print_fn = print # Changed: Always use print for logging

    if not BINANCE_CLIENT_AVAILABLE:
        _print_fn("CRITICAL ERROR in fetch_continuous_aggregate_trades: python-binance library not found."); sys.stdout.flush()
        return pd.DataFrame()

    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")
    cache_file = get_data_path_for_day(daily_file_date_str, symbol, data_type="agg_trades", cache_dir=cache_dir)

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
    if log_level in ["normal", "detailed"]:
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

    # Changed: Simplified tqdm initialization for a single bar
    local_fetch_pbar = tqdm(total=max(1, end_ms - current_start_ms),
                            desc=f"Fetching {symbol} for {daily_file_date_str}",
                            unit="ms", unit_scale=True, leave=False,
                            disable=(log_level == "none"),
                            mininterval=0.5 # Kept mininterval for smoother updates
                           )

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

# Alias for backward compatibility if needed, though direct usage of fetch_continuous_aggregate_trades is preferred.
fetch_and_cache_tick_data = fetch_continuous_aggregate_trades
