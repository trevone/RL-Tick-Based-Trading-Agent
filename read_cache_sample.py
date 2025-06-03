# read_cache_sample.py

import pandas as pd
import os
import sys

# Add the parent directory of this script to the Python path
# to ensure utils.py can be imported if it's in the project root.
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..')) # Assuming utils.py is in the project root
if project_root not in sys.path:
    sys.path.append(project_root)

# If utils.py is in the same directory as this script, you might not need the sys.path modification.
from utils import load_tick_data_for_range, DATA_CACHE_DIR

def read_cached_data_sample():
    """
    Reads cached tick data for a sample date range and prints its info.
    """
    sample_symbol = "BTCUSDT"
    sample_start_date = "2025-05-01"
    sample_end_date = "2025-05-02" # Dates for which you recently downloaded data

    print(f"Attempting to read cached data for {sample_symbol} from {sample_start_date} to {sample_end_date}...")
    print(f"Cache directory: {DATA_CACHE_DIR}")

    try:
        # load_tick_data_for_range will use the internal retry logic if it's in check_tick_cache.py
        # or handle missing files/empty data gracefully.
        cached_df = load_tick_data_for_range(
            symbol=sample_symbol,
            start_date_str=sample_start_date,
            end_date_str=sample_end_date,
            cache_dir=DATA_CACHE_DIR
        )

        if not cached_df.empty:
            print("\nSuccessfully loaded data from cache!")
            print(f"DataFrame Shape: {cached_df.shape}")
            print("DataFrame Head (first 5 rows):\n", cached_df.head())
            print("DataFrame Tail (last 5 rows):\n", cached_df.tail())
            print(f"Time range of loaded data: {cached_df.index.min()} to {cached_df.index.max()}")
        else:
            print("\nNo data was loaded from cache. The DataFrame is empty.")
            print("This could mean data was not found, or the files are still inaccessible.")

    except Exception as e:
        print(f"\nAn error occurred while trying to read cached data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    read_cached_data_sample()