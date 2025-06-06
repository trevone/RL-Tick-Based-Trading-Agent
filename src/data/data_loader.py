# src/data/data_loader.py
import os
import pandas as pd
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict
from tqdm import tqdm

from src.data.path_manager import get_data_path_for_day, _get_range_cache_path, DATA_CACHE_DIR
from src.data.binance_client import fetch_continuous_aggregate_trades, fetch_and_cache_kline_data

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

    pbar_days = tqdm(total=(end_date_obj - start_date_obj).days + 1, desc=f"Processing Ticks {symbol}", leave=True, disable=(log_level=="none"))

    while current_date_obj <= end_date_obj:
        date_str_for_day = current_date_obj.strftime('%Y-%m-%d')
        _print_fn = pbar_days.write

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
            except Exception as e:
                if log_level != "none": _print_fn(f"Error loading daily RESAMPLED file {daily_file_path}: {e}. Will try to generate from raw."); sys.stdout.flush()
                if os.path.exists(daily_file_path): os.remove(daily_file_path)
                df_daily = pd.DataFrame()
        
        if df_daily.empty:
            raw_daily_file_path = get_data_path_for_day(date_str_for_day, symbol, data_type="agg_trades", cache_dir=cache_dir)
            df_raw_daily = pd.DataFrame()

            if os.path.exists(raw_daily_file_path):
                try:
                    df_raw_daily = pd.read_parquet(raw_daily_file_path)
                    if df_raw_daily.index.tz is None and not df_raw_daily.empty: df_raw_daily.index = df_raw_daily.index.tz_localize('UTC')
                except Exception as e_load_raw:
                    if log_level != "none": _print_fn(f"  WARNING: Could not read existing RAW daily file {raw_daily_file_path} (Error: {e_load_raw}). Re-downloading."); sys.stdout.flush()
            
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
            
            if df_raw_daily.empty:
                current_date_obj += timedelta(days=1)
                pbar_days.update(1)
                continue

            if tick_resample_interval_ms:
                if not isinstance(df_raw_daily.index, pd.DatetimeIndex) or df_raw_daily.index.tz is None:
                     df_raw_daily.index = pd.to_datetime(df_raw_daily.index, utc=True)
                try:
                    freq_str = f"{tick_resample_interval_ms}ms"
                    agg_rules = {'Price': 'last', 'Quantity': 'sum', 'IsBuyerMaker': 'last'}
                    valid_agg_rules = {col: rule for col, rule in agg_rules.items() if col in df_raw_daily.columns}
                    df_daily = df_raw_daily.resample(freq_str).agg(valid_agg_rules)
                    df_daily.ffill(inplace=True); df_daily.bfill(inplace=True)
                    for col, default_val in {'Price': 0, 'Quantity': 0, 'IsBuyerMaker': False}.items():
                        if col not in df_daily.columns: df_daily[col] = default_val
                    df_daily.fillna({'Price': 0, 'Quantity': 0, 'IsBuyerMaker': False}, inplace=True)
                    os.makedirs(os.path.dirname(daily_file_path), exist_ok=True)
                    df_daily.to_parquet(daily_file_path)
                except Exception as e_resample:
                    if log_level != "none": _print_fn(f"Error resampling daily tick data for {date_str_for_day}: {e_resample}. Using raw."); sys.stdout.flush()
                    df_daily = df_raw_daily.copy()
            else:
                df_daily = df_raw_daily.copy()

        if not df_daily.empty:
            all_data_frames.append(df_daily)
        current_date_obj += timedelta(days=1)
        pbar_days.update(1)
    pbar_days.close()

    if not all_data_frames:
        if log_level != "none": print(f"No tick data found or loaded for {symbol} in range {start_date_str} to {end_date_str}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames)
    if not combined_df.empty:
        combined_df = combined_df.sort_index(); combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        try:
            start_datetime_utc = pd.to_datetime(start_date_str, utc=True)
            end_datetime_utc = pd.to_datetime(end_date_str, utc=True)
            original_count = len(combined_df)
            combined_df = combined_df.loc[start_datetime_utc:end_datetime_utc]
            if log_level in ["normal", "detailed"] and original_count > 0: print(f"Applied precise datetime filter: {original_count} -> {len(combined_df)} rows from {start_datetime_utc} to {end_datetime_utc}")
        except Exception as e_filter:
            if log_level != "none": print(f"WARNING: Could not apply precise datetime filter to combined tick data: {e_filter}.")
        
        try:
            os.makedirs(os.path.dirname(range_cache_file_path), exist_ok=True)
            if log_level != "none": print(f"Saving combined tick data to range cache: {range_cache_file_path}")
            combined_df.to_parquet(range_cache_file_path)
        except Exception as e:
            if log_level != "none": print(f"Error saving combined tick data to range cache {range_cache_file_path}: {e}")
    return combined_df

# ... (The `load_kline_data_for_range` function is also correct and needs no further changes)

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

    if log_level == "detailed":
        print(f"DEBUG_LOAD_KLINE (Range): Attempting to load from full range cache: {range_cache_file_path}"); sys.stdout.flush()

    if os.path.exists(range_cache_file_path):
        try:
            if log_level != "none": print(f"Loading FULL RANGE kline data from cache: {range_cache_file_path}"); sys.stdout.flush()
            df_combined = pd.read_parquet(range_cache_file_path)
            if not df_combined.empty and isinstance(df_combined.index, pd.DatetimeIndex):
                if df_combined.index.tz is None: df_combined.index = df_combined.index.tz_localize('UTC')
                if all(feat in df_combined.columns for feat in price_features):
                    if log_level == "detailed": print(f"DEBUG_LOAD_KLINE (Range): Full range cache HIT. Shape: {df_combined.shape}"); sys.stdout.flush()
                    return df_combined
                else:
                    if log_level != "none": print(f"Warning: Range cache file {range_cache_file_path} missing requested features. Re-processing."); sys.stdout.flush()
                    os.remove(range_cache_file_path)
            else:
                 if log_level != "none": print(f"Warning: Range cache file {range_cache_file_path} is empty or invalid. Re-processing."); sys.stdout.flush()
                 os.remove(range_cache_file_path)
        except Exception as e:
            if log_level != "none": print(f"Error loading from range cache {range_cache_file_path}: {e}. Re-processing."); sys.stdout.flush()
            if os.path.exists(range_cache_file_path): os.remove(range_cache_file_path)

    start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
    all_data_frames = []
    current_date_obj = start_date_obj

    pbar_days_kline = tqdm(total=(end_date_obj - start_date_obj).days + 1, desc=f"Processing Klines {symbol} {interval}", leave=True)

    while current_date_obj <= end_date_obj:
        date_str_for_day = current_date_obj.strftime('%Y-%m-%d')
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
            all_data_frames.append(df_daily)

        current_date_obj += timedelta(days=1)
        pbar_days_kline.update(1)
    pbar_days_kline.close()

    if not all_data_frames:
        if log_level != "none": print(f"No K-line data found or loaded for {symbol} in range {start_date_str} to {end_date_str} for interval {interval}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames)
    if not combined_df.empty:
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        try:
            start_datetime_utc = pd.to_datetime(start_date_str, utc=True)
            end_datetime_utc = pd.to_datetime(end_date_str, utc=True)
            original_count = len(combined_df)
            combined_df = combined_df.loc[start_datetime_utc:end_datetime_utc]
            if log_level in ["normal", "detailed"] and original_count > 0:
                print(f"Applied precise datetime filter: {original_count} -> {len(combined_df)} rows from {start_datetime_utc} to {end_datetime_utc}")
        except Exception as e_filter:
            if log_level != "none": print(f"WARNING: Could not apply precise datetime filter to combined k-line data: {e_filter}.")

        for feat in price_features:
            if feat not in combined_df.columns:
                if log_level != "none": print(f"Warning: Feature '{feat}' missing in combined kline data. Filling with 0.")
                combined_df[feat] = 0.0
        try:
            combined_df = combined_df[price_features]
        except KeyError as e:
            if log_level != "none": print(f"Error selecting final columns for K-line data: {e}. Available: {combined_df.columns.tolist()}")
            combined_df = combined_df[[col for col in price_features if col in combined_df.columns]]

        try:
            os.makedirs(os.path.dirname(range_cache_file_path), exist_ok=True)
            if log_level != "none": print(f"Saving combined kline data to range cache: {range_cache_file_path}")
            combined_df.to_parquet(range_cache_file_path)
        except Exception as e:
            if log_level != "none": print(f"Error saving combined kline data to range cache {range_cache_file_path}: {e}")

    return combined_df