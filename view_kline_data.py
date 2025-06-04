import pandas as pd
import os

def print_kline_samples(filepath: str, num_samples: int = 5):
    """
    Loads a K-line Parquet file and prints the first few samples.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    try:
        df = pd.read_parquet(filepath)
        
        if df.empty:
            print(f"The DataFrame loaded from '{filepath}' is empty.")
            return

        print(f"\n--- Samples from: {filepath} ---")
        print(f"DataFrame Shape: {df.shape}")
        print(f"DataFrame Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head(num_samples))
        print("\nLast few rows:")
        print(df.tail(num_samples))

    except Exception as e:
        print(f"Error loading or processing file '{filepath}': {e}")

if __name__ == "__main__":
    # Define the path to one of your cached K-line files.
    # Adjust this path if your file name or symbol/date differs.
    # Use the exact file name printed in your previous successful run.
    kline_filepath = "./binance_data_cache/bn_klines_BTCUSDT_1h_2025-05-01_to_2025-05-01_close_high_low_open_rsi14_sma20_volume.parquet"
    
    print_kline_samples(kline_filepath, num_samples=500)

    # You can also try another day's file if you downloaded multiple
    # kline_filepath_2 = "./binance_data_cache/bn_klines_BTCUSDT_1h_2025-05-02_to_2025-05-02_close_high_low_open_rsi14_sma20_volume.parquet"
    # print_kline_samples(kline_filepath_2, num_samples=5)