# verify_observation.py

import numpy as np
import pandas as pd
from src.data.config_loader import load_config
from src.data.data_loader import load_tick_data_for_range, load_kline_data_for_range

# NOTE: The following import assumes you have created the 'live candle' version of the environment.
# If you have it in a different file or class, please adjust the import accordingly.
# This example uses the corrected base_env logic we developed.
from src.environments.base_env import SimpleTradingEnv

def find_last_values():
    """
    Loads the environment for one step, gets the observation, and prints
    the key values for verification.
    """
    print("--- Loading Configuration and Data ---")
    
    # Load configuration from your provided YAML file
    config = load_config(main_config_path="config.yaml", default_config_paths=[
        "configs/defaults/run_settings.yaml",
        "configs/defaults/binance_settings.yaml",
        "configs/defaults/environment.yaml",
        "configs/defaults/ppo_params.yaml",
    ])
    
    run_settings = config["run_settings"]
    env_config = config["environment"]
    
    # Load just enough data to initialize the environment
    print("Loading tick data...")
    tick_df = load_tick_data_for_range(
        run_settings["default_symbol"], 
        run_settings["start_date_train"], 
        run_settings["end_date_train"], 
        run_settings["historical_cache_dir"], 
        config["binance_settings"], 
        env_config.get("tick_resample_interval_ms"),
        log_level="none"
    )
    
    print("Loading K-line data...")
    kline_df = load_kline_data_for_range(
        run_settings["default_symbol"], 
        run_settings["start_date_train"], 
        run_settings["end_date_train"], 
        run_settings["historical_interval"], 
        env_config["kline_price_features"], 
        run_settings["historical_cache_dir"], 
        config["binance_settings"],
        log_level="none"
    )

    if tick_df.empty or kline_df.empty:
        print("\nERROR: Could not load data. Cannot run verification.")
        return
        
    print("\n--- Initializing Environment ---")
    env = SimpleTradingEnv(tick_df=tick_df, kline_df_with_ta=kline_df, config=env_config)
    
    # Get the very first observation from the environment
    obs, info = env.reset()
    
    print("--- Verifying Observation Values ---")
    print(f"Current Step: {info.get('current_step')}")
    print(f"Current Timestamp: {info.get('timestamp')}")

    # --- 1. Get the ground truth tick price from the info dictionary ---
    last_tick_price_actual = info.get('current_tick_price')
    print(f"\nMethod 1: Actual last tick price (from info dict) = {last_tick_price_actual:.2f}")

    # --- 2. Calculate the index for the last candle's Close price ---
    tick_window = env_config['tick_feature_window_size']
    kline_window = env_config['kline_window_size']
    kline_features = env_config['kline_price_features']
    
    kline_block_start = len(env_config['tick_features_to_use']) * tick_window
    
    try:
        close_feature_index = kline_features.index('Close')
    except ValueError:
        print("ERROR: 'Close' is not in your kline_price_features in config.yaml")
        return
        
    last_close_price_index = kline_block_start + (close_feature_index * kline_window) + (kline_window - 1)
    
    # Extract the value from the observation array
    last_candle_close_from_obs = obs[last_close_price_index]
    print(f"Method 2: Last candle's 'Close' in observation   = {last_candle_close_from_obs:.2f} (at index {last_close_price_index})")
    
    # --- 3. Compare the values ---
    print("\n--- VERIFICATION RESULT ---")
    if np.isclose(last_tick_price_actual, last_candle_close_from_obs):
        print("✅ SUCCESS: The last candle's Close price matches the last tick price.")
    else:
        print("❌ FAILURE: The values do not match.")
        print("NOTE: This is expected if you have not yet implemented the 'live candle' logic in base_env.py.")

    # Uncomment the line below if you want to see the full observation array
    # print("\nFull Observation Array:\n", obs)


if __name__ == "__main__":
    find_last_values()