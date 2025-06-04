# src/data/data_downloader_manager.py

#python data_downloader_manager.py --start_date 2025-04-01 --end_date 2025-06-02 --symbol BTCUSDT
#python data_downloader_manager.py --start_date 2025-05-01 --end_date 2025-05-02 --symbol BTCUSDT --data_type kline --interval 1h --kline_features Close Volume

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import time # Import time module for sleep (still useful for general delays)
# Removed: from tqdm import tqdm # NEW: Import tqdm for progress bar

# Import the necessary functions from your updated utils.py and check_tick_cache.py
from src.data.utils import fetch_and_cache_tick_data, get_data_path_for_day, fetch_and_cache_kline_data, DATA_CACHE_DIR
from src.data.check_tick_cache import validate_daily_data # No explicit log_level here, but we'll pass it in the call

def download_and_manage_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str):
    """
    Downloads and manages tick data for a specified date range and symbol.
    Each day's data is saved in a separate parquet file if it doesn't exist,
    and then validated.
    """
    start_date_range = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date_range = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()

    total_days = (end_date_range - start_date_range).days + 1
    
    current_date = start_date_range
    # Removed: with tqdm(total=total_days, desc=f"Downloading {symbol} agg_trades", unit="day") as pbar:
    while current_date <= end_date_range:
        date_str_for_file = current_date.strftime('%Y-%m-%d')

        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(microseconds=1)

        start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        file_path = get_data_path_for_day(date_str_for_file, symbol, data_type="agg_trades")

        print(f"Processing {symbol} {date_str_for_file}") # Modified to print directly

        data_fetched = None # Initialize to None for the current day's processing
        
        # 1. Check if the parquet file for the day already exists and is not empty/corrupt
        if os.path.exists(file_path):
            try:
                temp_df = pd.read_parquet(file_path)
                if not temp_df.empty:
                    print(f"  File already exists: {file_path}. Skipping download.") # Modified to print directly
                    data_fetched = temp_df # Indicate data is available from existing file
                else:
                    print(f"  Warning: Existing file {file_path} is empty. Attempting to re-download.") # Modified to print directly
                    os.remove(file_path) # Remove empty file to force re-download
            except Exception as e:
                print(f"  Error reading existing file {file_path}: {e}. Attempting to re-download.") # Modified to print directly
                if os.path.exists(file_path):
                    os.remove(file_path) # Remove corrupt file to force re-download

        # If no valid existing file was found, attempt download
        if data_fetched is None or data_fetched.empty:
            print(f"  File not found or invalid. Attempting to download and cache data...") # Modified to print directly
            try:
                data_fetched = fetch_and_cache_tick_data(symbol, start_datetime_str_for_api, end_datetime_str_for_api)
                
                if data_fetched is not None and not data_fetched.empty:
                    print(f"  Successfully downloaded and cached data to {file_path}.") # Modified to print directly
                else:
                    print(f"  No data returned by API for {symbol} on {date_str_for_file} ({start_datetime_str_for_api} to {end_datetime_str_for_api}). File not created or is empty.") # Modified to print directly
                    current_date += timedelta(days=1)
                    # Removed: pbar.update(1) # Update progress even if no data for the day
                    continue # Skip validation if no data was fetched or file is empty

            except Exception as e:
                print(f"  Error downloading data for {symbol} on {date_str_for_file}: {e}") # Modified to print directly
                current_date += timedelta(days=1)
                # Removed: pbar.update(1) # Update progress even on error
                continue

        # 2. Proceed with validation ONLY if data_fetched is not None and not empty.
        if data_fetched is not None and not data_fetched.empty:
            print(f"  Validating data for {symbol} on {date_str_for_file}...") # Modified to print directly
            try:
                # MODIFIED: Pass log_level='none' to suppress verbose validation output
                is_valid, message = validate_daily_data(file_path, log_level='none') # MODIFIED
                if is_valid:
                    print(f"  Validation successful: {message}") # Modified to print directly
                else:
                    print(f"  Validation failed: {message}") # Modified to print directly
            except Exception as e:
                print(f"  Error during validation for {symbol} on {date_str_for_file}: {e}") # Modified to print directly
        else:
            print(f"  Validation skipped for {symbol} on {date_str_for_file} as no valid DataFrame was obtained.") # Modified to print directly

        current_date += timedelta(days=1)
        # Removed: pbar.update(1) # Update progress for the current day
    print("\nAggregate trades data management process completed.")


def download_and_manage_kline_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str, interval: str, price_features_to_add: list):
    """
    Downloads and manages K-line data for a specified date range, symbol, interval,
    and optional technical indicators. Each day's data is saved in a separate parquet file if it doesn't exist,
    and then validated.
    """
    start_date_range = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date_range = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()

    total_days = (end_date_range - start_date_range).days + 1

    current_date = start_date_range
    # Removed: with tqdm(total=total_days, desc=f"Downloading {symbol} klines ({interval})", unit="day") as pbar:
    while current_date <= end_date_range:
        date_str_for_file = current_date.strftime('%Y-%m-%d')

        # For Klines, start and end time within the day
        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc) # Max time for kline end

        start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S") # Klines API uses full second resolution

        file_path = get_data_path_for_day(date_str_for_file, symbol, data_type="kline", 
                                        interval=interval, price_features_to_add=price_features_to_add)

        print(f"Processing {symbol} {date_str_for_file} {interval}") # Modified to print directly

        data_fetched = None
        
        # 1. Check if the parquet file for the day already exists and is not empty/corrupt
        if os.path.exists(file_path):
            try:
                temp_df = pd.read_parquet(file_path)
                if not temp_df.empty:
                    print(f"  File already exists: {file_path}. Skipping download.") # Modified to print directly
                    data_fetched = temp_df
                else:
                    print(f"  Warning: Existing file {file_path} is empty. Attempting to re-download.") # Modified to print directly
                    os.remove(file_path)
            except Exception as e:
                print(f"  Error reading existing file {file_path}: {e}. Attempting to re-download.") # Modified to print directly
                if os.path.exists(file_path):
                    os.remove(file_path)

        # If no valid existing file was found, attempt download
        if data_fetched is None or data_fetched.empty:
            print(f"  File not found or invalid. Attempting to download and cache K-line data...") # Modified to print directly
            try:
                data_fetched = fetch_and_cache_kline_data(
                    symbol=symbol,
                    interval=interval,
                    start_date_str=start_datetime_str_for_api,
                    end_date_str=end_datetime_str_for_api,
                    cache_dir=DATA_CACHE_DIR, # Use DATA_CACHE_DIR from src.data.utils
                    price_features_to_add=price_features_to_add
                )
                
                if data_fetched is not None and not data_fetched.empty:
                    print(f"  Successfully downloaded and cached K-line data to {file_path}.") # Modified to print directly
                else:
                    print(f"  No K-line data returned by API for {symbol} on {date_str_for_file} for interval {interval}. File not created or is empty.") # Modified to print directly
                    current_date += timedelta(days=1)
                    # Removed: pbar.update(1) # Update progress even if no data for the day
                    continue
            except Exception as e:
                print(f"  Error downloading K-line data for {symbol} on {date_str_for_file}: {e}") # Modified to print directly
                current_date += timedelta(days=1)
                # Removed: pbar.update(1) # Update progress even on error
                continue

        # 2. Proceed with validation ONLY if data_fetched is not None and not empty.
        if data_fetched is not None and not data_fetched.empty:
            print(f"  Validating K-line data for {symbol} on {date_str_for_file}...") # Modified to print directly
            try:
                # MODIFIED: Pass log_level='none' to suppress verbose validation output
                is_valid, message = validate_daily_data(file_path, log_level='none') # MODIFIED
                if is_valid:
                    print(f"  Validation successful: {message}") # Modified to print directly
                else:
                    print(f"  Validation failed: {message}") # Modified to print directly
            except Exception as e:
                print(f"  Error during validation for {symbol} on {date_str_for_file}: {e}") # Modified to print directly
        else:
            print(f"  Validation skipped for {symbol} on {date_str_for_file} as no valid DataFrame was obtained.") # Modified to print directly

        current_date += timedelta(days=1)
        # Removed: pbar.update(1) # Update progress for the current day
    print("\nK-line data management process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and manage data for a specified date range.")
    parser.add_argument("--start_date", required=True, help="Start date in Walpole-MM-DD format.")
    parser.add_argument("--end_date", required=True, help="End date in Walpole-MM-DD format.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., 'EURUSD' or 'BTCUSDT').")
    parser.add_argument("--data_type", default="agg_trades", choices=["agg_trades", "kline"], 
                        help="Type of data to download: 'agg_trades' or 'kline'. Default: agg_trades")
    parser.add_argument("--interval", default="1h", help="Interval for kline data (e.g., '1m', '1h', '1d'). Only applies to 'kline' data_type. Default: 1h")
    parser.add_argument("--kline_features", nargs='*', default=['Open', 'High', 'Low', 'Close', 'Volume'], 
                        help="Space-separated list of kline price features (e.g., 'Open High Close SMA_20'). Only applies to 'kline' data_type.")

    args = parser.parse_args()

    # NOTE: Using current date for future date checks.
    # Current time: Wednesday, June 4, 2025 at 6:30:55 PM BST.
    if datetime.strptime(args.start_date, '%Y-%m-%d').date() > datetime(2025, 6, 4).date():
        print("\nWARNING: Your start date is in the future. Binance API provides historical data, "
              "not future data. You will likely receive no data.\n")
    elif datetime.strptime(args.end_date, '%Y-%m-%d').date() > datetime(2025, 6, 4).date():
         print("\nWARNING: Your end date is in the future. Binance API provides historical data up to current time, "
              "so you might receive partial data for the end date, or no data if start is also future.\n")

    if args.data_type == "agg_trades":
        download_and_manage_data(args.start_date, args.end_date, args.symbol)
    elif args.data_type == "kline":
        # Ensure that kline_features are appropriately handled if you load them from config later
        download_and_manage_kline_data(args.start_date, args.end_date, args.symbol, args.interval, args.kline_features)
    else:
        print(f"Error: Invalid data_type '{args.data_type}' specified.")