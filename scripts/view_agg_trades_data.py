# scripts/view_agg_trades_data.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- UPDATED IMPORTS ---
from src.data.binance_client import fetch_continuous_aggregate_trades
from src.data.path_manager import DATA_CACHE_DIR
# --- END UPDATED IMPORTS ---

def main():
    parser = argparse.ArgumentParser(description="View Binance Aggregate Trade data from cache or fetch.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (e.g., BTCUSDT).")
    parser.add_argument("--start_date", default="2024-01-01 00:00:00", 
                        help="Start date and time in 'YYYY-MM-DD HH:MM:SS' format.")
    parser.add_argument("--end_date", default="2024-01-01 00:05:00", 
                        help="End date and time in 'YYYY-MM-DD HH:MM:SS' format.")
    parser.add_argument("--cache_dir", default=DATA_CACHE_DIR, # Use the imported DATA_CACHE_DIR
                        help=f"Directory for cached data. Default: {DATA_CACHE_DIR}")
    
    args = parser.parse_args()

    print(f"Fetching Aggregate Trade data for {args.symbol} from {args.start_date} to {args.end_date}...")

    df_trades = fetch_continuous_aggregate_trades(
        symbol=args.symbol,
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        cache_dir=args.cache_dir,
        log_level="normal" # Use 'normal' for interactive script
    )

    if df_trades.empty:
        print("No aggregate trade data retrieved. Exiting.")
        return

    print(f"\n--- Aggregate Trades Data Head ({args.symbol}) ---")
    print(df_trades.head())
    print(f"\n--- Aggregate Trades Data Tail ({args.symbol}) ---")
    print(df_trades.tail())
    print(f"\nDataFrame shape: {df_trades.shape}")
    print(f"Time range: {df_trades.index.min()} to {df_trades.index.max()}")
    print(f"Columns: {df_trades.columns.tolist()}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot Price over time
    ax1.plot(df_trades.index, df_trades['Price'], label='Price', color='blue', linewidth=0.5)
    ax1.set_title(f'{args.symbol} Aggregate Trades - Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # Plot Quantity over time (or Volume)
    # Sum quantity for each second if there are many ticks to avoid clutter
    df_trades_resampled_qty = df_trades['Quantity'].resample('1S').sum().replace(0, np.nan) # Sum per second, replace 0 with NaN for gaps
    ax2.bar(df_trades_resampled_qty.index, df_trades_resampled_qty.values, width=1/24/60/60, # 1 second width
            color='green', alpha=0.7, label='Quantity per Second')
    ax2.set_title(f'{args.symbol} Aggregate Trades - Quantity')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Quantity')
    ax2.legend()
    ax2.grid(True)

    # Format x-axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()