# scripts/read_cache_sample.py

import pandas as pd
import os
import argparse
import sys
import traceback

# --- UPDATED IMPORTS ---
from src.data.path_manager import get_data_path_for_day, DATA_CACHE_DIR
# --- END UPDATED IMPORTS ---

def main():
    parser = argparse.ArgumentParser(description="Read and display a sample of cached data.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (e.g., BTCUSDT).")
    parser.add_argument("--date", default="2024-01-01", help="Date in YYYY-MM-DD format.")
    parser.add_argument("--data_type", default="agg_trades", choices=["agg_trades", "kline"], 
                        help="Type of data to load: 'agg_trades' or 'kline'.")
    parser.add_argument("--interval", default="1h", help="Interval for kline data (e.g., 1m, 1h, 1d). Required for 'kline' data_type.")
    parser.add_argument("--features", nargs='*', 
                        default=['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI_14'],
                        help="List of kline price features (e.g., Open Close SMA_20). Only for 'kline' data_type for filename lookup.")
    parser.add_argument("--cache_dir", default=DATA_CACHE_DIR, # Use the imported DATA_CACHE_DIR
                        help=f"Directory where data is cached. Default: {DATA_CACHE_DIR}")
    
    args = parser.parse_args()

    if args.data_type == "kline" and not args.interval:
        print("Error: --interval is required for 'kline' data_type.")
        sys.exit(1)

    try:
        file_path = get_data_path_for_day(
            date_str=args.date,
            symbol=args.symbol,
            data_type=args.data_type,
            interval=args.interval if args.data_type == "kline" else None,
            price_features_to_add=args.features if args.data_type == "kline" else None,
            cache_dir=args.cache_dir
        )

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            print("Please ensure the data has been downloaded and cached using data_manager.py.")
            sys.exit(1)

        print(f"Attempting to read data from: {file_path}")
        df = pd.read_parquet(file_path)

        if df.empty:
            print("The DataFrame is empty.")
            return

        print(f"\n--- DataFrame Head ---")
        print(df.head())
        print(f"\n--- DataFrame Tail ---")
        print(df.tail())
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Index type: {type(df.index)}")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"Index timezone: {df.index.tz}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()