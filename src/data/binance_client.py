# src/data/binance_client.py
import os
import pandas as pd
import time
import sys
import traceback
from tqdm import tqdm
from src.data.feature_engineer import calculate_technical_indicators
from src.data.path_manager import get_data_path_for_day

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False

def fetch_and_cache_kline_data(
    symbol: str, interval: str, start_date_str: str, end_date_str: str,
    cache_dir: str,
    price_features_to_add: dict = None,
    api_key: str = None, api_secret: str = None, testnet: bool = False,
    cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2, pbar_instance = None
) -> pd.DataFrame:

    _print_fn = print

    if not BINANCE_CLIENT_AVAILABLE:
        _print_fn("CRITICAL ERROR in fetch_and_cache_kline_data: python-binance library not found.")
        return pd.DataFrame()

    # The logic for creating the filename is now centralized in path_manager
    daily_file_date_str = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")
    cache_file = get_data_path_for_day(
        date_str=daily_file_date_str, symbol=symbol, data_type="kline",
        interval=interval, price_features_to_add=price_features_to_add, cache_dir=cache_dir
    )

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if log_level == "detailed":
        _print_fn(f"DEBUG_KLINE_DAILY: Checking for daily K-line cache: {cache_file}"); sys.stdout.flush()

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: _print_fn(f"Loading K-line data from daily cache: {cache_file}"); sys.stdout.flush()
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')

            feature_names = []
            if price_features_to_add:
                if isinstance(price_features_to_add, dict):
                    feature_names = list(price_features_to_add.keys())
                else:
                    feature_names = price_features_to_add

            missing_tas = [ta for ta in feature_names if ta not in df.columns]
            if missing_tas:
                if log_level != "none": _print_fn(f"Warning: Cached K-line data {cache_file} missing TAs: {missing_tas}. Refetching for this day."); sys.stdout.flush()
                os.remove(cache_file)
            else:
                if log_level == "detailed": _print_fn(f"DEBUG_KLINE_DAILY: Daily K-line cache HIT and valid: {cache_file}, Shape: {df.shape}"); sys.stdout.flush()
                return df
        except Exception as e:
            if log_level != "none": _print_fn(f"Error loading K-line data from daily cache {cache_file}: {e}. Refetching for this day."); sys.stdout.flush()
            if os.path.exists(cache_file): os.remove(cache_file)

    if log_level in ["normal", "detailed"]:
        _print_fn(f"Fetching K-line data for {symbol}, Interval: {interval}, Date: {daily_file_date_str}"); sys.stdout.flush()

    client = Client(api_key or os.environ.get('BINANCE_API_KEY'),
                    api_secret or os.environ.get('BINANCE_API_SECRET'),
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    try:
        klines_raw = client.get_historical_klines(symbol, interval, start_date_str, end_str=end_date_str)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        if price_features_to_add:
            if isinstance(price_features_to_add, dict):
                feature_names.extend([k for k in price_features_to_add.keys() if k not in feature_names])
            else:
                feature_names.extend([f for f in price_features_to_add if f not in feature_names])

        if not klines_raw:
            if log_level != "none": _print_fn(f"No k-lines returned by API for {symbol} {start_date_str}-{end_date_str}."); sys.stdout.flush()
            empty_df = pd.DataFrame(columns=feature_names)
            empty_df.index = pd.to_datetime([]).tz_localize('UTC')
            if cache_file_type == "parquet": empty_df.to_parquet(cache_file)
            else: empty_df.to_csv(cache_file)
            return empty_df

        kline_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume',
                        'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore']
        df_fetched = pd.DataFrame(klines_raw, columns=kline_cols)
        df_fetched['OpenTimeDate'] = pd.to_datetime(df_fetched['OpenTime'], unit='ms', utc=True)
        df_fetched.set_index('OpenTimeDate', inplace=True)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df_fetched.columns: df_fetched[col] = pd.to_numeric(df_fetched[col], errors='coerce')
        
        df_to_process = df_fetched[numeric_cols].copy().astype(float)
        
        # Pass the dictionary directly to the feature engineer
        df_with_ta = calculate_technical_indicators(df_to_process, price_features_to_add or {})

        if not df_with_ta.empty:
            try:
                if cache_file_type == "parquet": df_with_ta.to_parquet(cache_file)
                else: df_with_ta.to_csv(cache_file)
                if log_level in ["normal", "detailed"]: _print_fn(f"Daily K-line data saved to cache: {cache_file}"); sys.stdout.flush()
            except Exception as e: _print_fn(f"Error saving daily K-line data to cache {cache_file}: {e}"); sys.stdout.flush()
        return df_with_ta
        
    except BinanceAPIException as bae:
        _print_fn(f"Binance API Exception: Code={bae.code}, Message='{bae.message}'"); sys.stdout.flush()
    except Exception as e:
        _print_fn(f"Unexpected error: {e}"); sys.stdout.flush()
        traceback.print_exc()

    return pd.DataFrame(columns=feature_names).set_index(pd.to_datetime([]).tz_localize('UTC'))

def fetch_continuous_aggregate_trades(
    symbol: str, start_date_str: str, end_date_str: str,
    cache_dir: str, api_key: str = None, api_secret: str = None,
    testnet: bool = False, cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2, pbar_instance = None
) -> pd.DataFrame:

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

fetch_and_cache_tick_data = fetch_continuous_aggregate_trades