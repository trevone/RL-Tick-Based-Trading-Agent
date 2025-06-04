# view_kline_data.py
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import from the new utils path
from src.data.utils import fetch_and_cache_kline_data, DATA_CACHE_DIR

def main():
    parser = argparse.ArgumentParser(description="View Binance K-line data from cache or fetch.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (e.g., BTCUSDT).")
    parser.add_argument("--interval", default="1h", help="Interval for kline data (e.g., 1m, 1h, 1d).")
    parser.add_argument("--start_date", default="2024-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", default="2024-01-03", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--features", nargs='*', 
                        default=['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI_14', 'MACD', 'ATR'],
                        help="Space-separated list of K-line price features to display (e.g., Open Close SMA_20).")
    parser.add_argument("--cache_dir", default=DATA_CACHE_DIR, # Use the imported DATA_CACHE_DIR
                        help=f"Directory for cached data. Default: {DATA_CACHE_DIR}")
    
    args = parser.parse_args()

    print(f"Fetching K-line data for {args.symbol}, interval {args.interval} from {args.start_date} to {args.end_date}...")

    df_klines = fetch_and_cache_kline_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        cache_dir=args.cache_dir,
        price_features_to_add=args.features,
        log_level="normal" # Use 'normal' for interactive script
    )

    if df_klines.empty:
        print("No K-line data retrieved. Exiting.")
        return

    print(f"\n--- K-line Data Head ({args.symbol}, {args.interval}) ---")
    print(df_klines.head())
    print(f"\n--- K-line Data Tail ({args.symbol}, {args.interval}) ---")
    print(df_klines.tail())
    print(f"\nDataFrame shape: {df_klines.shape}")
    print(f"Time range: {df_klines.index.min()} to {df_klines.index.max()}")
    print(f"Columns: {df_klines.columns.tolist()}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot Close Price
    ax.plot(df_klines.index, df_klines['Close'], label='Close Price', color='blue', linewidth=1.5)

    # Plot requested features if they exist
    for feature in args.features:
        if feature in df_klines.columns and feature not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            ax.plot(df_klines.index, df_klines[feature], label=feature, linestyle='--', linewidth=0.8)
        
        # Specific handling for OHLCV to avoid replotting 'Close'
        elif feature in ['Open', 'High', 'Low', 'Volume']:
            # You could plot these differently, e.g., using candlestick if desired.
            # For simplicity, just skip to avoid cluttering if only 'Close' is needed
            pass

    ax.set_title(f'{args.symbol} {args.interval} K-line Data with Features')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()