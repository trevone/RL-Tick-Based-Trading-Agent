# data_downloader_manager.py

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import time # Import time module for sleep (still useful for general delays)

# Import the necessary functions from your updated utils.py and check_tick_cache.py
from utils import fetch_and_cache_tick_data, get_data_path_for_day
from check_tick_cache import validate_daily_data

def download_and_manage_data(start_date_str_arg: str, end_date_str_arg: str, symbol: str):
    """
    Downloads and manages tick data for a specified date range and symbol.
    Each day's data is saved in a separate parquet file if it doesn't exist,
    and then validated.
    """
    start_date_range = datetime.strptime(start_date_str_arg, '%Y-%m-%d').date()
    end_date_range = datetime.strptime(end_date_str_arg, '%Y-%m-%d').date()

    current_date = start_date_range
    while current_date <= end_date_range:
        date_str_for_file = current_date.strftime('%Y-%m-%d')

        day_start_dt_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end_dt_utc = datetime.combine(current_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(microseconds=1)

        start_datetime_str_for_api = day_start_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str_for_api = day_end_dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        file_path = get_data_path_for_day(date_str_for_file, symbol, data_type="agg_trades")

        print(f"Processing data for {symbol} on {date_str_for_file} (API range: {start_datetime_str_for_api} to {end_datetime_str_for_api})...")

        data_fetched = None # Initialize to None for the current day's processing
        file_was_downloaded_this_run = False # Flag to track if we just downloaded data

        # --- LOGGING: Before initial file check ---
        print(f"  DEBUG: Expected file_path for current day: {file_path}")
        print(f"  DEBUG: os.path.exists(file_path) before check: {os.path.exists(file_path)}")


        # 1. Check if the parquet file for the day already exists and is not empty/corrupt
        if os.path.exists(file_path):
            try:
                temp_df = pd.read_parquet(file_path)
                if not temp_df.empty:
                    print(f"  File already exists: {file_path}. Skipping download.")
                    data_fetched = temp_df # Indicate data is available from existing file
                else:
                    print(f"  Warning: Existing file {file_path} is empty. Attempting to re-download.")
                    os.remove(file_path) # Remove empty file to force re-download
            except Exception as e:
                print(f"  Error reading existing file {file_path}: {e}. Attempting to re-download.")
                if os.path.exists(file_path):
                    os.remove(file_path) # Remove corrupt file to force re-download

        # If no valid existing file was found, attempt download
        if data_fetched is None or data_fetched.empty:
            print(f"  File not found or invalid. Attempting to download and cache data...")
            try:
                data_fetched = fetch_and_cache_tick_data(symbol, start_datetime_str_for_api, end_datetime_str_for_api)
                
                if data_fetched is not None and not data_fetched.empty:
                    print(f"  Successfully downloaded and cached data to {file_path}.")
                    file_was_downloaded_this_run = True # Set flag because we just downloaded
                else:
                    print(f"  No data returned by API for {symbol} on {date_str_for_file} ({start_datetime_str_for_api} to {end_datetime_str_for_api}). File not created or is empty.")
                    current_date += timedelta(days=1)
                    continue # Skip validation if no data was fetched or file is empty

            except Exception as e:
                print(f"  Error downloading data for {symbol} on {date_str_for_file}: {e}")
                current_date += timedelta(days=1)
                continue

        # --- LOGGING: Before validation check ---
        print(f"  DEBUG: Before validation check (after download/load attempt):")
        print(f"  DEBUG:   data_fetched is not None: {data_fetched is not None}")
        print(f"  DEBUG:   data_fetched is not empty: {not data_fetched.empty if data_fetched is not None else False}")
        # Re-check os.path.exists only for debug, but validation logic will not rely on it
        print(f"  DEBUG:   os.path.exists(file_path) (just before validation): {os.path.exists(file_path)}")


        # 2. Proceed with validation ONLY if data_fetched is not None and not empty.
        #    We assume if data_fetched is populated, the file should be there and readable.
        if data_fetched is not None and not data_fetched.empty:
            print(f"  Validating data for {symbol} on {date_str_for_file}...")
            try:
                # Use validate_daily_data directly. If file reading fails, it will be caught here.
                is_valid, message = validate_daily_data(file_path)
                if is_valid:
                    print(f"  Validation successful: {message}")
                else:
                    print(f"  Validation failed: {message}")
            except Exception as e:
                print(f"  Error during validation for {symbol} on {date_str_for_file}: {e}")
                # If validation itself fails, it might mean the file is corrupt or not fully written
                # even if data_fetched was populated. We can optionally remove it here.
                # if os.path.exists(file_path):
                #     print(f"  Removing potentially corrupt file: {file_path}")
                #     os.remove(file_path)
        else:
            # This 'else' block implies data_fetched was None or empty,
            # which means no valid data was obtained for the day.
            print(f"  Validation skipped for {symbol} on {date_str_for_file} as no valid DataFrame was obtained.")


        current_date += timedelta(days=1)
    print("\nData management process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and manage tick data for a specified date range.")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., 'EURUSD' or 'BTCUSDT').")

    args = parser.parse_args()

    if datetime.strptime(args.start_date, '%Y-%m-%d').date() > datetime.now().date():
        print("\nWARNING: Your start date is in the future. Binance API provides historical data, "
              "not future data. You will likely receive no data.\n")
    elif datetime.strptime(args.end_date, '%Y-%m-%d').date() > datetime.now().date():
         print("\nWARNING: Your end date is in the future. Binance API provides historical data up to current time, "
              "so you might receive partial data for the end date, or no data if start is also future.\n")

    download_and_manage_data(args.start_date, args.end_date, args.symbol)