# src/data/data_manager.py

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys
import traceback

from src.data.config_loader import load_config
from src.data.path_manager import get_data_path_for_day
from src.data.binance_client import fetch_and_cache_tick_data, fetch_and_cache_kline_data
from src.data.data_validator import validate_daily_data

LOG_DIR_BASE_NAME = "logs"
DATA_MANAGEMENT_LOG_FILENAME = "data_management.log"

def _get_project_root():
    """Gets the project root directory based on this script's location."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _ensure_log_file_path():
    """Ensures the log directory exists and returns the full log file path."""
    project_root = _get_project_root()
    log_dir_full_path = os.path.join(project_root, LOG_DIR_BASE_NAME)
    os.makedirs(log_dir_full_path, exist_ok=True)
    return os.path.join(log_dir_full_path, DATA_MANAGEMENT_LOG_FILENAME)

def _log_deletion_event(file_path: str, reason: str):
    """Logs the deletion of a cached file with improved formatting."""
    log_file_full_path = _ensure_log_file_path()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    reason_summary = reason
    details_section = []

    validation_failed_prefix = "Validation failed: "
    if reason.startswith(validation_failed_prefix):
        reason_summary = "Validation Failed"
        original_validation_message = reason[len(validation_failed_prefix):].strip()
        specific_errors_header = "One or more checks FAILED:"
        if original_validation_message.startswith(specific_errors_header):
            error_lines_str = original_validation_message[len(specific_errors_header):].strip()
            details_section = [f"  {line.strip()}" for line in error_lines_str.splitlines() if line.strip()]
        else:
            details_section = [f"  {original_validation_message}"]
    
    log_entry_parts = [
        "--------------------------------------------------",
        f"Timestamp: {timestamp}",
        "Event:     DELETED CACHED FILE",
        f"File Path: {file_path}",
        f"Reason:    {reason_summary}"
    ]
    
    if details_section:
        log_entry_parts.append("Details:")
        log_entry_parts.extend(details_section)
    elif reason_summary != reason:
        log_entry_parts.append("Details:")
        log_entry_parts.append(f"  {reason}")

    log_entry_parts.append("--------------------------------------------------\n")
    log_message = "\n".join(log_entry_parts)

    try:
        with open(log_file_full_path, "a", encoding="utf-8") as f:
            f.write(log_message)
    except Exception as e:
        print(f"  ERROR: Failed to write to deletion log ({log_file_full_path}): {e}")

def load_configs_for_data_management(config_dir="configs/defaults") -> dict:
    """Loads configurations for data management tasks."""
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "binance_settings.yaml")
    ]
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)

def _manage_single_day_data(date_str_for_file, symbol, data_type, historical_cache_dir, path_manager_kwargs, fetch_function, fetch_function_kwargs):
    """
    Generic helper function to manage downloading and validating data for a single day.
    """
    print(f"\n--- Processing {symbol} {data_type} for {date_str_for_file} ---")
    file_path = get_data_path_for_day(date_str_for_file, symbol, data_type=data_type, cache_dir=historical_cache_dir, **path_manager_kwargs)
    sys.stdout.flush()

    data_fetched = None
    
    if os.path.exists(file_path):
        try:
            if os.path.getsize(file_path) == 0:
                reason = "File was 0 bytes."
                print(f"  Warning: Existing file {file_path} is 0 bytes. Deleting.")
                os.remove(file_path)
                _log_deletion_event(file_path, reason)
            else:
                temp_df = pd.read_parquet(file_path)
                if not temp_df.empty:
                    print(f"  File already exists and is not empty: {file_path}.")
                    data_fetched = temp_df
                else:
                    reason = "File was empty after reading."
                    print(f"  Warning: Existing file {file_path} is empty after reading. Deleting.")
                    os.remove(file_path)
                    _log_deletion_event(file_path, reason)
        except Exception as e:
            reason = f"File unreadable/corrupt: {e}"
            print(f"  Error reading existing file {file_path}: {e}. Deleting.")
            if os.path.exists(file_path):
                os.remove(file_path)
                _log_deletion_event(file_path, reason)

    if data_fetched is None:
        print(f"  Attempting to download and cache data...")
        sys.stdout.flush()
        try:
            # FIX: Pass 'normal' log_level to activate the progress bar in the client
            fetch_function_kwargs['log_level'] = 'normal'
            data_fetched = fetch_function(**fetch_function_kwargs)
            if data_fetched is not None and not data_fetched.empty:
                print(f"  Successfully downloaded and cached data to {file_path}.")
            else:
                print(f"  No data returned by API for this period.")
        except Exception as e:
            print(f"  Error during download: {e}")
            traceback.print_exc()

    if data_fetched is not None and not data_fetched.empty:
        if not os.path.exists(file_path):
            print(f"  Validation skipped as file {file_path} does not exist (download might have failed).")
        else:
            print(f"  Validating data at {file_path}...")
            sys.stdout.flush()
            try:
                is_valid, message = validate_daily_data(file_path, log_level='none')
                if is_valid:
                    print(f"  Validation successful: {message.splitlines()[0]}")
                else:
                    reason = f"Validation failed: {message}"
                    print(f"  VALIDATION FAILED: {message}")
                    os.remove(file_path)
                    print(f"  DELETED invalid cached file: {file_path}")
                    _log_deletion_event(file_path, reason)
            except Exception as e_val:
                print(f"  Error during validation: {e_val}")
    else:
        print(f"  Validation skipped as no valid data was obtained.")
    sys.stdout.flush()

def download_and_manage_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str):
    """
    Manages downloading and validating aggregate trade data for a date range.
    """
    effective_config = load_configs_for_data_management()
    run_settings = effective_config.get("run_settings", {})
    binance_settings = effective_config.get("binance_settings", {})
    historical_cache_dir = run_settings.get("historical_cache_dir", "data_cache/")
    
    print(f"--- Managing AGGREGATE TRADES ---")
    print(f"Using cache directory: {os.path.abspath(historical_cache_dir)}")
    
    current_date = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc)
        
        fetch_kwargs = {
            "symbol": symbol,
            "start_date_str": day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date_str": day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "cache_dir": historical_cache_dir,
            "api_key": binance_settings.get("api_key"),
            "api_secret": binance_settings.get("api_secret"),
            "testnet": binance_settings.get("testnet", False),
        }

        _manage_single_day_data(
            date_str_for_file=date_str,
            symbol=symbol,
            data_type="agg_trades",
            historical_cache_dir=historical_cache_dir,
            path_manager_kwargs={},
            fetch_function=fetch_and_cache_tick_data,
            fetch_function_kwargs=fetch_kwargs
        )
        current_date += timedelta(days=1)
    print("\nAggregate trades data management process completed.")

def download_and_manage_kline_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str, interval: str, price_features_to_add: list):
    """
    Manages downloading and validating K-line data for a date range.
    """
    effective_config = load_configs_for_data_management()
    run_settings = effective_config.get("run_settings", {})
    binance_settings = effective_config.get("binance_settings", {})
    historical_cache_dir = run_settings.get("historical_cache_dir", "data_cache/")

    print(f"--- Managing K-LINE DATA ({interval}) ---")
    print(f"Using cache directory: {os.path.abspath(historical_cache_dir)}")

    current_date = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc)

        fetch_kwargs = {
            "symbol": symbol,
            "interval": interval,
            "start_date_str": day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date_str": day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_dir": historical_cache_dir,
            "price_features_to_add": price_features_to_add,
            "api_key": binance_settings.get("api_key"),
            "api_secret": binance_settings.get("api_secret"),
            "testnet": binance_settings.get("testnet", False),
        }
        path_kwargs = {
            "interval": interval,
            "price_features_to_add": price_features_to_add
        }

        _manage_single_day_data(
            date_str_for_file=date_str,
            symbol=symbol,
            data_type="kline",
            historical_cache_dir=historical_cache_dir,
            path_manager_kwargs=path_kwargs,
            fetch_function=fetch_and_cache_kline_data,
            fetch_function_kwargs=fetch_kwargs
        )
        current_date += timedelta(days=1)
    print("\nK-line data management process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and manage data for a specified date range.")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., 'BTCUSDT').")
    parser.add_argument("--data_type", default="agg_trades", choices=["agg_trades", "kline"], 
                        help="Type of data to download: 'agg_trades' or 'kline'. Default: agg_trades")
    parser.add_argument("--interval", default="1h", help="Interval for kline data (e.g., '1m', '1h', '1d'). Only applies to 'kline' data_type. Default: 1h")
    parser.add_argument("--kline_features", nargs='*', default=['Open', 'High', 'Low', 'Close', 'Volume'], 
                        help="Space-separated list of kline price features (e.g., 'Open High Close SMA_20'). Only applies to 'kline' data_type.")
    
    args = parser.parse_args()

    current_system_date = datetime.now(timezone.utc).date() 
    start_date_arg_dt = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date_arg_dt = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    if start_date_arg_dt > current_system_date:
        print(f"\nWARNING: Your start date ({args.start_date}) is in the future. You will likely receive no data.\n")
    elif end_date_arg_dt > current_system_date:
         print(f"\nWARNING: Your end date ({args.end_date}) is in the future. You may receive partial data.\n")
    if end_date_arg_dt < start_date_arg_dt:
        print(f"\nERROR: Your end date ({args.end_date}) is before your start date ({args.start_date}).\n")
        sys.exit(1)

    if args.data_type == "agg_trades":
        download_and_manage_data(args.start_date, args.end_date, args.symbol)
    elif args.data_type == "kline":
        download_and_manage_kline_data(args.start_date, args.end_date, args.symbol, args.interval, args.kline_features)
    else:
        print(f"Error: Invalid data_type '{args.data_type}' specified.")