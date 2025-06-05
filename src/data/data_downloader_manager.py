# src/data/data_downloader_manager.py

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.data.utils import fetch_and_cache_tick_data, get_data_path_for_day, fetch_and_cache_kline_data, DATA_CACHE_DIR
from src.data.check_tick_cache import validate_daily_data

# --- Logging for Deleted Files ---
LOG_DIR_BASE_NAME = "logs"  # Name of the logs directory
DATA_MANAGEMENT_LOG_FILENAME = "data_management.log"

def _get_project_root():
    """Gets the project root directory based on this script's location."""
    # Assumes this script is in ProjectRoot/src/data/
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
    details_section = [] # Will hold formatted detail lines

    validation_failed_prefix = "Validation failed: "
    if reason.startswith(validation_failed_prefix):
        reason_summary = "Validation Failed"
        # Extract the original message part from validate_daily_data
        original_validation_message = reason[len(validation_failed_prefix):].strip()
        
        # Check if it's the detailed "One or more checks FAILED:" message
        specific_errors_header = "One or more checks FAILED:"
        if original_validation_message.startswith(specific_errors_header):
            # Get lines after the header, these are the specific error messages
            error_lines_str = original_validation_message[len(specific_errors_header):].strip()
            # Format each error line for the details section
            details_section = [f"  {line.strip()}" for line in error_lines_str.splitlines() if line.strip()]
        else:
            # It's a simpler validation failure message (e.g., "Could not read Parquet file...")
            details_section = [f"  {original_validation_message}"]
    
    # Construct the log message
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
    elif reason_summary != reason: # Should only happen if we summarized but didn't find specific details structure
        log_entry_parts.append("Details:")
        log_entry_parts.append(f"  {reason}")


    log_entry_parts.append("--------------------------------------------------\n") # Add a newline at the end
    
    log_message = "\n".join(log_entry_parts)

    try:
        with open(log_file_full_path, "a", encoding="utf-8") as f:
            f.write(log_message)
    except Exception as e:
        print(f"  ERROR: Failed to write to deletion log ({log_file_full_path}): {e}")
# --- END Logging for Deleted Files ---

def download_and_manage_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str):
    start_date_range = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date_range = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()
    
    current_date = start_date_range
    while current_date <= end_date_range:
        date_str_for_file = current_date.strftime('%Y-%m-%d')

        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(microseconds=1)

        start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        file_path = get_data_path_for_day(date_str_for_file, symbol, data_type="agg_trades", cache_dir=DATA_CACHE_DIR)

        print(f"Processing {symbol} agg_trades for {date_str_for_file}")

        data_fetched = None 
        
        if os.path.exists(file_path):
            try:
                if os.path.getsize(file_path) == 0:
                    reason_for_del = "File was 0 bytes."
                    print(f"  Warning: Existing file {file_path} is 0 bytes. Deleting and re-downloading.")
                    os.remove(file_path)
                    _log_deletion_event(file_path, reason_for_del)
                else:
                    temp_df = pd.read_parquet(file_path)
                    if not temp_df.empty:
                        print(f"  File already exists and is not empty: {file_path}. Proceeding to validation.")
                        data_fetched = temp_df 
                    else:
                        reason_for_del = "File was empty after reading."
                        print(f"  Warning: Existing file {file_path} is empty after reading. Deleting and re-downloading.")
                        os.remove(file_path)
                        _log_deletion_event(file_path, reason_for_del)
            except Exception as e: 
                reason_for_del = f"File unreadable/corrupt: {e}"
                print(f"  Error reading existing file {file_path}: {e}. Deleting and re-downloading.")
                if os.path.exists(file_path): 
                    os.remove(file_path)
                    _log_deletion_event(file_path, reason_for_del)

        if data_fetched is None: 
            print(f"  Attempting to download and cache data for {date_str_for_file}...")
            try:
                data_fetched = fetch_and_cache_tick_data(symbol, start_datetime_str_for_api, end_datetime_str_for_api, cache_dir=DATA_CACHE_DIR, log_level='normal')
                
                if data_fetched is not None and not data_fetched.empty:
                    print(f"  Successfully downloaded and cached data to {file_path}.")
                else:
                    print(f"  No data returned by API for {symbol} on {date_str_for_file}. File not created or is empty.")
            except Exception as e:
                print(f"  Error downloading data for {symbol} on {date_str_for_file}: {e}")

        if data_fetched is not None and not data_fetched.empty:
            if not os.path.exists(file_path):
                 print(f"  Validation skipped for {symbol} on {date_str_for_file} as file {file_path} does not exist (download might have failed silently).")
            else:
                print(f"  Validating data for {symbol} on {date_str_for_file} (Path: {file_path})...")
                try:
                    is_valid, message = validate_daily_data(file_path, log_level='none') 
                    if is_valid:
                        print(f"  Validation successful: {message.splitlines()[0]}")
                    else:
                        reason_for_del = f"Validation failed: {message}"
                        print(f"  VALIDATION FAILED for {file_path}: {message}")
                        try:
                            os.remove(file_path)
                            print(f"  DELETED invalid cached file: {file_path}")
                            _log_deletion_event(file_path, reason_for_del)
                        except OSError as e_del:
                            print(f"  ERROR: Could not delete invalid file {file_path}: {e_del}")
                except Exception as e_val:
                    print(f"  Error during validation for {symbol} on {date_str_for_file}: {e_val}")
        else:
            print(f"  Validation skipped for {symbol} on {date_str_for_file} as no valid data was obtained or downloaded.")

        current_date += timedelta(days=1)
    print("\nAggregate trades data management process completed.")


def download_and_manage_kline_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str, interval: str, price_features_to_add: list):
    start_date_range = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date_range = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()

    current_date = start_date_range
    while current_date <= end_date_range:
        date_str_for_file = current_date.strftime('%Y-%m-%d')

        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc) 

        start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S") 

        file_path = get_data_path_for_day(date_str_for_file, symbol, data_type="kline", 
                                        interval=interval, price_features_to_add=price_features_to_add,
                                        cache_dir=DATA_CACHE_DIR)

        print(f"Processing {symbol} klines ({interval}) for {date_str_for_file}")

        data_fetched = None
        
        if os.path.exists(file_path):
            try:
                if os.path.getsize(file_path) == 0:
                    reason_for_del = "File was 0 bytes."
                    print(f"  Warning: Existing file {file_path} is 0 bytes. Deleting and re-downloading.")
                    os.remove(file_path)
                    _log_deletion_event(file_path, reason_for_del)
                else:
                    temp_df = pd.read_parquet(file_path)
                    if not temp_df.empty:
                        print(f"  File already exists and is not empty: {file_path}. Proceeding to validation.")
                        data_fetched = temp_df
                    else:
                        reason_for_del = "File was empty after reading."
                        print(f"  Warning: Existing file {file_path} is empty after reading. Deleting and re-downloading.")
                        os.remove(file_path)
                        _log_deletion_event(file_path, reason_for_del)
            except Exception as e:
                reason_for_del = f"File unreadable/corrupt: {e}"
                print(f"  Error reading existing file {file_path}: {e}. Deleting and re-downloading.")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    _log_deletion_event(file_path, reason_for_del)

        if data_fetched is None:
            print(f"  Attempting to download and cache K-line data for {date_str_for_file}...")
            try:
                data_fetched = fetch_and_cache_kline_data(
                    symbol=symbol,
                    interval=interval,
                    start_date_str=start_datetime_str_for_api,
                    end_date_str=end_datetime_str_for_api,
                    cache_dir=DATA_CACHE_DIR,
                    price_features_to_add=price_features_to_add,
                    log_level='normal'
                )
                
                if data_fetched is not None and not data_fetched.empty:
                    print(f"  Successfully downloaded and cached K-line data to {file_path}.")
                else:
                    print(f"  No K-line data returned by API for {symbol} on {date_str_for_file} for interval {interval}.")
            except Exception as e:
                print(f"  Error downloading K-line data for {symbol} on {date_str_for_file}: {e}")

        if data_fetched is not None and not data_fetched.empty:
            if not os.path.exists(file_path):
                 print(f"  Validation skipped for Klines {symbol} on {date_str_for_file} as file {file_path} does not exist.")
            else:
                print(f"  Validating K-line data for {symbol} on {date_str_for_file} (Path: {file_path})...")
                try:
                    is_valid, message = validate_daily_data(file_path, log_level='none') 
                    if is_valid:
                        print(f"  Validation successful: {message.splitlines()[0]}")
                    else:
                        reason_for_del = f"Validation failed: {message}"
                        print(f"  VALIDATION FAILED for {file_path}: {message}")
                        try:
                            os.remove(file_path)
                            print(f"  DELETED invalid cached file: {file_path}")
                            _log_deletion_event(file_path, reason_for_del)
                        except OSError as e_del:
                            print(f"  ERROR: Could not delete invalid file {file_path}: {e_del}")
                except Exception as e_val:
                    print(f"  Error during validation for Klines {symbol} on {date_str_for_file}: {e_val}")
        else:
            print(f"  Validation skipped for Klines {symbol} on {date_str_for_file} as no valid data was obtained or downloaded.")

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
    if datetime.strptime(args.start_date, '%Y-%m-%d').date() > current_system_date:
        print(f"\nWARNING: Your start date ({args.start_date}) is in the future (current date: {current_system_date.strftime('%Y-%m-%d')}). "
              "Binance API provides historical data, not future data. You will likely receive no data.\n")
    elif datetime.strptime(args.end_date, '%Y-%m-%d').date() > current_system_date:
         print(f"\nWARNING: Your end date ({args.end_date}) is in the future (current date: {current_system_date.strftime('%Y-%m-%d')}). "
               "Binance API provides historical data up to current time. "
               "You might receive partial data for the end date, or no data if start is also future.\n")

    if args.data_type == "agg_trades":
        download_and_manage_data(args.start_date, args.end_date, args.symbol)
    elif args.data_type == "kline":
        download_and_manage_kline_data(args.start_date, args.end_date, args.symbol, args.interval, args.kline_features)
    else:
        print(f"Error: Invalid data_type '{args.data_type}' specified.")